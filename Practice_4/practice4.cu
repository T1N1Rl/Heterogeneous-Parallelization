#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace chrono;

__global__ void reductionSharedKernel(int* d_in, int* d_out, int n) { // __global__ функция вызывается с CPU, а выполняется на GPU (ядро)
    extern __shared__ int sdata[];                  // Объявление динамической разделяемой памяти

    int tid = threadIdx.x;                          // Локальный номер потока внутри блока (0-255)
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Глобальный индекс элемента в массиве

    // Копирование данных из Глобальной памяти в Разделяемую память
    sdata[tid] = (i < n) ? d_in[i] : 0;             // Если индекс за пределами массива, кладем 0 (нейтральный элемент для суммы)
    __syncthreads();                                // Синхронизация внутри блока

    // Редукция внутри блока в Разделяемой памяти
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {  // Мы используем алгоритм "дерева", сокращая количество работающих потоков вдвое на каждом шаге
        if (tid < s) {
            sdata[tid] += sdata[tid + s];           // Поток суммирует два элемента в Разделяемой памяти
        }
        __syncthreads();                            // Ждем завершения текущего шага суммирования всем блоком
    }

    // Запись результата в Глобальную память
    if (tid == 0) d_out[blockIdx.x] = sdata[0];     // Только нулевой поток блока записывает результат (сумму всех элементов блока)
}


__global__ void sortSharedKernel(int* d_arr, int n) {  // Ядро для сортировки маленьких кусков массива (подмассивов)
    __shared__ int s_part[256];                        // Фиксированный размер разделяемой памяти для одного блока (256 элементов)

    int tid = threadIdx.x;                             // Локальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // Глобальный индекс

    // Загрузка подмассива из Глобальной памяти в Разделяемую память
    if (idx < n) s_part[tid] = d_arr[idx];
    else s_part[tid] = 2147483647;                     // Заполняем пустоты максимальным числом (уйдет в конец при сортировке)
    __syncthreads();                                   // Ждем, пока весь блок загрузит свои данные в Разделяемую память

    // Сортировка пузырьком выполняется внутри Разделяемой памяти (Локальная оптимизация)
    if (tid == 0) {                                   // Сортирует один поток для простоты понимания оптимизации памяти
        int size = min(256, n - (int)(blockIdx.x * blockDim.x));   // Определяем реальный размер куска
        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < size - i - 1; j++) {
                if (s_part[j] > s_part[j + 1]) {
                    int temp = s_part[j];     //  Обмен через временную Локальную переменную
                    s_part[j] = s_part[j + 1];
                    s_part[j + 1] = temp;
                }
            }
        }
    }
    __syncthreads();                         // Ждем, пока поток №0 закончит сортировку всего Shared-буфера

    // Выгрузка отсортированного подмассива обратно в Глобальную память
    if (idx < n) d_arr[idx] = s_part[tid];
}

void benchmark(int N) {
    cout << "\n>>> Test size: " << N << endl; // Вывод текущего размера тестируемого массива
    
    vector<int> h_arr(N);       // Выделение массива в оперативной памяти CPU (host)
    long long cpu_sum = 0;      // Переменная для вычисления суммы на процессоре
    for (int i = 0; i < N; i++) {  // Заполнение массива случайными числами
        h_arr[i] = rand() % 10; // Ограничиваем числа от 0 до 9, чтобы сумма не переполнилась быстро
        cpu_sum += h_arr[i];    // Считаем контрольную сумму на CPU
    }

    int *d_arr;          // Указатель на основной массив в видеопамяти
    int *d_out;          // Указатель на массив результатов редукции в видеопамяти
    int blockSize = 256; // Устанавливаем размер блока (256 потоков)
    int numBlocks = (N + blockSize - 1) / blockSize;   // Считаем количество необходимых блоков

    cudaMalloc(&d_arr, N * sizeof(int));   // Выделение памяти на GPU (Device)
    cudaMalloc(&d_out, numBlocks * sizeof(int));
    cudaMemcpy(d_arr, h_arr.data(), N * sizeof(int), cudaMemcpyHostToDevice);  // Копирование исходного массива с CPU на GPU

    // Редукция (Разделяемая)
    auto start = high_resolution_clock::now();   // Засекаем старт
    reductionSharedKernel<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_arr, d_out, N);  // Запуск ядра
    cudaDeviceSynchronize();                     // Остановка CPU до полного завершения вычислений на GPU
    auto end = high_resolution_clock::now();     // Засекаем конец

    // Копирование частичных сумм обратно на CPU для финального сложения
    vector<int> h_out(numBlocks);                 // Вектор для хранения частичных сумм с GPU
    cudaMemcpy(h_out.data(), d_out, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    long long gpu_sum = 0;
    for (int x : h_out) gpu_sum += x;            // Складываем результаты от каждого блока

    // Вывод результатов редукции
    cout << "Reduction (Shared): " << duration<double, milli>(end - start).count() << " ms" << endl;
    cout << "Result: " << (gpu_sum == cpu_sum ? "Correct" : "Error") << " (Sum: " << gpu_sum << ")" << endl;

    // Сортировка (Разделяемая + Локальная)
    cudaMemcpy(d_arr, h_arr.data(), N * sizeof(int), cudaMemcpyHostToDevice); // Сброс
    start = high_resolution_clock::now();        // Старт замера времени сортировки
    sortSharedKernel<<<numBlocks, blockSize>>>(d_arr, N);  // Запуск ядра сортировки подмассивов
    cudaDeviceSynchronize();                     // Ждем окончания
    end = high_resolution_clock::now();          // Конец замера
    cout << "Sorting subarrays (Shared Memory): " << duration<double, milli>(end - start).count() << " ms" << endl;

    // Очистка памяти на видеокарте
    cudaFree(d_arr);
    cudaFree(d_out);
}

int main() {   // Запуск тестов для трех размеров массива согласно заданию
    benchmark(10000);   // Тест на 10000 элементов
    benchmark(100000);  // Тест на 100000 элементов
    benchmark(1000000); // Тест на 1000000 элементов
    return 0;           // Завершение программы
}