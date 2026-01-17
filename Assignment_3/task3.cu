#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

// 1. Ядро для коалесцированного доступа
__global__ void coalesced_copy(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Вычисляем глобальный индекс потока
    if (idx < n) {            // Если индекс внутри границ массива
        out[idx] = in[idx];   // Коалесцированный доступ: соседние потоки читают соседние ячейки
    }
}

// 2. Ядро для некоалесцированного доступа
__global__ void uncoalesced_copy(float* out, const float* in, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Вычисляем глобальный индекс потока
    
    // Вычисляем "разбросанный" индекс: умножаем ID потока на шаг (stride)
    // Операция % n гарантирует, что мы останемся в пределах массива
    int uncoalesced_idx = (idx * stride) % n;
    
    if (idx < n) {   // Если поток в пределах N
        // Некоалесцированный доступ: соседние потоки прыгают по памяти далеко друг от друга
        // Это заставляет контроллер памяти GPU делать много лишних запросов
        out[idx] = in[uncoalesced_idx];
    }
}

int main() {
    const int N = 1000000; // размер массива: 1000000 элементов
    const size_t size = N * sizeof(float);  // Рассчитываем размер массива в байтах
    const int stride = 32; // Шаг для некоалесцированного доступа

    // Подготовка данных на процессоре (хосте)
    vector<float> h_in(N, 1.0f); // Создаем массив и заполняем его единицами
    float *d_in, *d_out;         // Объявляем указатели для памяти видеокарты

    // Выделение памяти на видеокарте
    cudaMalloc(&d_in, size);    // Выделяем память под входной массив
    cudaMalloc(&d_out, size);   // Выделяем память под выходной массив
    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);   // Копируем данные из оперативной памяти (h_in) в видеопамять (d_in)

    int blockSize = 256;       // Количество потоков в одном блоке
    int gridSize = (N + blockSize - 1) / blockSize;  // Рассчитываем количество блоков, чтобы охватить все N элементов

    // Создаем события CUDA для высокоточного замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);  // Инициализируем событие начала
    cudaEventCreate(&stop);   // Инициализируем событие конца

    // Первый тест: Коалесцированный доступ
    cudaEventRecord(start);     // Фиксируем время старта первого теста
    // Запускаем быстрое ядро
    coalesced_copy<<<gridSize, blockSize>>>(d_out, d_in, N);
    cudaEventRecord(stop);      // Фиксируем время окончания первого теста
    cudaEventSynchronize(stop); // Ждем, пока GPU закончит выполнение всех операций
    float timeCoalesced = 0;    
    // Считаем разницу между метками старта и финиша в миллисекундах
    cudaEventElapsedTime(&timeCoalesced, start, stop);

    // Второй тест: Некоалесцированный доступ
    cudaEventRecord(start);     // Фиксируем время старта второго теста
    uncoalesced_copy<<<gridSize, blockSize>>>(d_out, d_in, N, stride);
    // Запускаем медленное ядро  
    cudaEventRecord(stop);      // Фиксируем время окончания второго теста
    cudaEventSynchronize(stop); // Снова ждем завершения работы GPU
    float timeUncoalesced = 0;
    // Считаем время для второго теста
    cudaEventElapsedTime(&timeUncoalesced, start, stop);

    // Вывод результатов
    cout << fixed << setprecision(6); // Настройка вывода 6 знаков после запятой
    cout << "Array size: " << N << " elements" << endl;
    cout << "Coalesced Access Time  : " << timeCoalesced << " ms" << endl;
    cout << "Uncoalesced Access Time: " << timeUncoalesced << " ms" << endl;
    // Во сколько раз некоалесцированный доступ медленнее
    cout << "Performance Drop       : " << timeUncoalesced / timeCoalesced << "x slower" << endl;

    // Очистка ресурсов
    cudaFree(d_in);            // Удаляем входной массив из видеопамяти
    cudaFree(d_out);           // Удаляем выходной массив из видеопамяти
    cudaEventDestroy(start);   // Удаляем объекты событий
    cudaEventDestroy(stop); 

    return 0;                 // Завершение программы
}