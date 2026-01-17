#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// Ядро с использованием только Глобальной памяти
__global__ void multiply_global(float* data, float factor, int n) {
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверка границ массива
    if (idx < n) {
        data[idx] = data[idx] * factor;   // Прямое чтение и запись в глобальную память (медленный способ из-за долгого доступа к VRAM)
    }
}

// Ядро с использованием Разделяемой памяти
__global__ void multiply_shared(float* data, float factor, int n) {
    // Объявляем динамическую разделяемую память
    extern __shared__ float temp[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Индекс в глобальном массиве
    int tid = threadIdx.x;                           // Индекс внутри текущего блока

    // Шаг 1: Загрузка данных из Глобальной памяти в быструю Разделяемую
    if (idx < n) {
        temp[tid] = data[idx];                       // Каждый поток загружает один элемент
    } else {
        temp[tid] = 0.0f;                            // Если массив закончился, заполняем нулем (безопасность)
    }

    // Синхронизация: ждем, пока все потоки блока закончат загрузку в temp[]
    __syncthreads();

    // Шаг 2: Обработка данных внутри Разделяемой памяти
    if (idx < n) {
        temp[tid] *= factor;                        // Математическая операция в shared memory  
    }

    // Синхронизация: ждем завершения вычислений перед выгрузкой
    __syncthreads();

    // Шаг 3: Запись результата обратно в Глобальную память
    if (idx < n) {
        data[idx] = temp[tid];
    }
}

int main() {
    const int N = 1000000;           // Размер массива
    const float factor = 2.0f;       // Число для умножения
    size_t size = N * sizeof(float); // Общий объем памяти в байтах

    // Подготовка данных на хосте (CPU)
    vector<float> h_data(N, 1.0f);  // Создаем массив на 1000000 единиц
    float *d_data;                  // Указатель на память видеокарты
    cudaMalloc(&d_data, size);      // Выделение памяти в Глобальной памяти GPU

    // Настройка сетки: 256 потоков в блоке
    int blockSize = 256;                             // Количество потоков в одном блоке
    int gridSize = (N + blockSize - 1) / blockSize;  // Расчет количества блоков для покрытия всего N

    // Создание событий для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);        // Создание объекта начала замера
    cudaEventCreate(&stop);         // Создание объекта конца замера

    // Первый тест: Глобальная память 
    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);   // Копируем исходные данные с CPU на GPU
    
    cudaEventRecord(start);                                       // Фиксируем время начала
    multiply_global<<<gridSize, blockSize>>>(d_data, factor, N);  // Запуск ядра (сетка блоков, потоков в блоке)   
    cudaEventRecord(stop);                                        // Фиксируем время окончания
    
    cudaEventSynchronize(stop);                                   // Ждем, пока GPU закончит работу
    float timeGlobal = 0;
    cudaEventElapsedTime(&timeGlobal, start, stop);               // Считаем разницу в мс

    // Второй тест: Разделяемая память
    // Сброс данных к исходным значениям
    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);   // Фиксируем время начала                                                                  
    // Третий параметр <<<...>>> — объем динамической Разделяемой памяти в байтах
    multiply_shared<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_data, factor, N);
    cudaEventRecord(stop);   // Фиксируем время окончания

    cudaEventSynchronize(stop);                      // Ожидание завершения видеокарты
    float timeShared = 0;
    cudaEventElapsedTime(&timeShared, start, stop);  // Считаем разницу в мс

    // Вывод результатов
    cout << "Array size: " << N << " elements" << endl;
    cout << "Global Memory Time: " << timeGlobal << " ms" << endl;
    cout << "Shared Memory Time: " << timeShared << " ms" << endl;

    // Очистка ресурсов
    cudaFree(d_data);           // Освобождение видеопамяти
    cudaEventDestroy(start);    // Удаление объектов замера времени
    cudaEventDestroy(stop);

    return 0;                  // Завершение программы
}