#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

// CUDA-ядро для поэлементного сложения
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    // Вычисляем глобальный индекс элемента
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка, чтобы не выйти за пределы массива 
    if (i < n) {
        C[i] = A[i] + B[i];   // Само сложение
    }
}

// Функция для запуска теста с конкретным размером блока
void runTest(int N, int blockSize) {
    size_t size = N * sizeof(float);  // Объем памяти для одного массива в байтах
 
    // Количество повторов для набора статистики
    const int iters = 1000;

    // Выделение памяти на хосте
    vector<float> h_A(N, 1.0f);  // Выделяем память под массив A
    vector<float> h_B(N, 2.0f);  // Выделяем память под массив B
    vector<float> h_C(N, 0.0f);  // Выделяем память под результат C

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Копирование данных на GPU
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Расчет количества блоков
    int gridSize = (N + blockSize - 1) / blockSize;

    // Создание событий для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Принудительно ждем завершения всех предыдущих операций на GPU
    cudaDeviceSynchronize();

    // Запуск ядра несколько раз и замер суммарного времени
    cudaEventRecord(start);
    for (int it = 0; it < iters; ++it) {                       // Цикл для многократного запуска ядра
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);  // Запуск ядра с текущими параметрами сетки
    }
    cudaEventRecord(stop);       // Записываем временную метку финиша

    cudaEventSynchronize(stop);  // Ждем, пока GPU физически дойдет до метки stop
    float totalMs = 0.0f;
    cudaEventElapsedTime(&totalMs, start, stop);  // Вычисляем разницу между метками start и stop в миллисекундах

    float perKernelMs = totalMs / iters;     // Считаем среднее время одного запуска

    // Вывод строки таблицы с форматированием
    cout << "| " << setw(10) << blockSize   // Размер блока
         << " | " << setw(10) << gridSize   // Размер сетки
         << " | " << setw(8)  << iters      // Количество итераций
         << " | " << setw(12) << fixed << setprecision(4) << totalMs << " ms"      // Общее время
         << " | " << setw(12) << fixed << setprecision(6) << perKernelMs << " ms"  // Среднее время
         << " |" << endl;

    // Очистка ресурсов
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int N = 10000000; // 10000000 элементов

    // Вывод заголовка таблицы
    cout << "Testing vector addition with N = " << N << endl;
    cout << "-----------------------------------------------------------------------------------------------" << endl;
    cout << "| Block Size | Grid Size  |   Iters  | Total Time     | Per-kernel Time |" << endl;
    cout << "-----------------------------------------------------------------------------------------------" << endl;

    // Массив разных размеров блока для исследования
    int sizes[] = {64, 256, 1024};
    // Перебираем размеры и запускаем тесты
    for (int s : sizes) {
        runTest(N, s);
    }

    cout << "-----------------------------------------------------------------------------------------------" << endl;
    return 0;   // Завершение программы
}
