#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

// CUDA-ядро для сложения векторов
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    // Вычисляем глобальный индекс потока
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Проверка, чтобы индекс не вышел за пределы массива
    if (i < n) {
        C[i] = A[i] + B[i];  // Сложение двух чисел
    }
}

// Функция для проведения замера
float measurePerformance(int N, int blockSize) {
    size_t size = N * sizeof(float);  // Рассчитываем объем памяти в байтах
    float *d_A, *d_B, *d_C;           // Указатели для видеопамяти

    // Выделение памяти
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Рассчитываем количество блоков (Grid), чтобы покрыть все N элементов
    int gridSize = (N + blockSize - 1) / blockSize;

    // События CUDA для высокоточного замера времени работы GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Замер времени
    cudaEventRecord(start);
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N); // Запуск ядра с указанной конфигурацией сетки и блоков
    cudaEventRecord(stop);                                // Фиксируем временную метку окончания
    cudaEventSynchronize(stop);                           // Ждем, пока GPU физически дойдет до метки стоп

    float ms = 0;
    // Считаем разницу между метками старта и стопа в миллисекундах
    cudaEventElapsedTime(&ms, start, stop);

    // Очистка 
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);      // Освобождение видеопамяти
    cudaEventDestroy(start); cudaEventDestroy(stop);  // Удаляем объекты событий

    return ms;   // Возвращаем время выполнения в мс
}

int main() {
    // Увеличим размер до 20 млн элементов, чтобы разница была видна отчетливо
    const int N = 20000000; 
    
    cout << fixed << setprecision(6);  // Настройка вывода: 6 знаков после запятой
    cout << "Performance Optimization Analysis (N = " << N << ")" << endl;
    cout << "------------------------------------------------------" << endl;

    // 1. Неоптимальная конфигурация (Маленький размер блока)
    int badBlockSize = 32;
    float timeBad = measurePerformance(N, badBlockSize);
    cout << "Non-optimal (Block Size: " << badBlockSize << ")  : " << timeBad << " ms" << endl;

    // 2. Оптимизированная конфигурация (Средний размер блока)
    int optimalBlockSize = 256;
    float timeOptimal = measurePerformance(N, optimalBlockSize);
    cout << "Optimized   (Block Size: " << optimalBlockSize << ") : " << timeOptimal << " ms" << endl;

    // 3. Крайняя конфигурация (Максимальный размер блока)
    int maxBlockSize = 1024;
    float timeMax = measurePerformance(N, maxBlockSize);
    cout << "Edge Case   (Block Size: " << maxBlockSize << ") : " << timeMax << " ms" << endl;

    // Считаем коэффициент ускорения
    cout << "------------------------------------------------------" << endl;
    cout << "Speedup (Optimized vs Non-optimal): " << timeBad / timeOptimal << "x" << endl;

    return 0; //Завершение программы
}