#include <iostream>                    ъ
#include <vector>                       
#include <algorithm>                    
#include <chrono>                       
#include <cuda_runtime.h>               
#include <device_launch_parameters.h>   

using namespace std;
using namespace chrono;


// Merge Sort CPU
void mergeCPU(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;                   // Размер левого подмассива
    int n2 = r - m;                       // Размер правого подмассива
    vector<int> L(n1);                    // Временный массив для левой части
    vector<int> R(n2);                    // Временный массив для правой части
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];  // Копируем левую часть
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j]; // Копируем правую часть
    int i = 0, j = 0, k = l;              // Индексы для слияния
    while (i < n1 && j < n2) {            // Слияние
        if (L[i] <= R[j]) arr[k++] = L[i++]; // Берем меньший элемент из левого
        else arr[k++] = R[j++];           // Или из правого
    }
    while (i < n1) arr[k++] = L[i++];     // Копируем остаток левой части
    while (j < n2) arr[k++] = R[j++];     // Копируем остаток правой части
}

void mergeSortCPU(vector<int>& arr, int l, int r) {
    if (l < r) {                          // Условие выхода из рекурсии
        int m = l + (r - l) / 2;          // Находим середину
        mergeSortCPU(arr, l, m);          // Сортируем левую половину
        mergeSortCPU(arr, m + 1, r);      // Сортируем правую половину
        mergeCPU(arr, l, m, r);           // Сливаем две половины
    }
}

// Quick Sort CPU
void quickSortCPU(vector<int>& arr, int l, int r) {
    if (l < r) {
        int pivot = arr[r];                // Опорный элемент
        int i = l - 1;                     // Индекс для разделения
        for (int j = l; j < r; j++)        // Проходим по массиву
            if (arr[j] < pivot) swap(arr[++i], arr[j]); // Перестановка
        swap(arr[i + 1], arr[r]);          // Размещение pivot
        quickSortCPU(arr, l, i);           // Рекурсия для левой части
        quickSortCPU(arr, i + 2, r);       // Рекурсия для правой части
    }
}

// Heap Sort CPU
void heapify(vector<int>& arr, int n, int i) {
    int largest = i;                       // Корень
    int l = 2 * i + 1;                     // Левый потомок
    int r = 2 * i + 2;                     // Правый потомок
    if (l < n && arr[l] > arr[largest]) largest = l; // Если левый потомок больше
    if (r < n && arr[r] > arr[largest]) largest = r; // Если правый потомок больше
    if (largest != i) {                    // Если корень не самый большой
        swap(arr[i], arr[largest]);        // Меняем местами
        heapify(arr, n, largest);          // Рекурсивно проверяем затронутое поддерево
    }
}

void heapSortCPU(vector<int>& arr) {
    int n = arr.size();
    for (int i = n / 2 - 1; i >= 0; i--) heapify(arr, n, i); // Построение кучи
    for (int i = n - 1; i > 0; i--) {    // Извлечение элементов
        swap(arr[0], arr[i]);            // Перемещаем текущий корень в конец
        heapify(arr, i, 0);              // Вызываем heapify на уменьшенной куче
    }
}



// Merge Sort GPU kernel
__global__ void mergeKernel(int* arr, int* temp, int width, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // Глобальный ID потока
    int start = id * 2 * width;                      // Начало сегмента
    if (start >= n) return;                          // Проверка границ
    int mid = min(start + width, n);                 // Середина рабочего участка
    int end = min(start + 2 * width, n);             // Конец рабочего участка
    int i = start, j = mid, k = start;               // Слияние
    while (i < mid && j < end)
        temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++]; // Записываем во временный массив
    while (i < mid) temp[k++] = arr[i++];            // Копируем остаток левой части
    while (j < end) temp[k++] = arr[j++];            // Копируем остаток правой части
}

// Quick Sort GPU kernel 
__device__ void deviceSwap(int& a, int& b) { int t = a; a = b; b = t; }  // Вспомогательная функция замены для устройства
__global__ void quickSortKernel(int* arr, int n, int segment) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // Глобальный ID потока
    int start = id * segment;                        // Начало сегмента
    int end = min(start + segment - 1, n - 1);       // Конец сегмента
    if (start >= end) return; 
    for (int i = start; i <= end; i++)               // Пузырьковая/простая сортировка внутри
        for (int j = i + 1; j <= end; j++)
            if (arr[i] > arr[j]) deviceSwap(arr[i], arr[j]);
}

// Heap Sort GPU kernel 
__global__ void heapSortKernel(int* arr, int n, int segment) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // Глобальный ID потока
    int start = id * segment;                        // Начало сегмента
    int end = min(start + segment, n);               // Конец сегмента
    if (start >= end) return;
    for (int i = start; i < end; i++)                // Сортировка сегмента одним потоком
        for (int j = i + 1; j < end; j++)
            if (arr[i] > arr[j]) deviceSwap(arr[i], arr[j]);
}


void benchmark(int N) {
    cout << "\n=== Array size: " << N << " ===\n";
    vector<int> h(N);                             // Создаем массив на хосте    
    for (int& x : h) x = rand();                  // Генерация случайных чисел

    int *d, *temp;
    cudaMalloc(&d, N * sizeof(int));             // Память GPU для массива
    cudaMalloc(&temp, N * sizeof(int));          // Временный буфер для Merge

    // --- CPU Merge Sort ---
    vector<int> arrCPU = h;                      // Копируем исходные данные
    auto start = high_resolution_clock::now();   // Засекаем время
    mergeSortCPU(arrCPU, 0, N - 1);              // Запуск сортировки 
    auto end = high_resolution_clock::now();     // Конец замера
    cout << "CPU Merge Sort: " << duration<double, milli>(end - start).count() << " ms\n";

    // --- CPU Quick Sort ---
    arrCPU = h;                                  // Копируем исходные данные
    start = high_resolution_clock::now();        // Засекаем время
    quickSortCPU(arrCPU, 0, N - 1);              // Запуск сортировки 
    end = high_resolution_clock::now();          // Конец замера
    cout << "CPU Quick Sort: " << duration<double, milli>(end - start).count() << " ms\n";

    // --- CPU Heap Sort ---
    arrCPU = h;                                  // Копируем исходные данные
    start = high_resolution_clock::now();        // Засекаем время
    heapSortCPU(arrCPU);                         // Запуск сортировки 
    end = high_resolution_clock::now();          // Конец замера
    cout << "CPU Heap Sort: " << duration<double, milli>(end - start).count() << " ms\n";

    // --- GPU Merge Sort ---
    cudaMemcpy(d, h.data(), N * sizeof(int), cudaMemcpyHostToDevice); // Копирование исходных данных на GPU
    start = high_resolution_clock::now();                             // Старт замера времени GPU
    for (int w = 1; w < N; w *= 2) {                                  // Итеративно увеличиваем ширину слияния
        int numThreads = 256;                                         // Количество потоков в блоке
        int numBlocks = (N + 2 * w * numThreads - 1) / (2 * w * numThreads); // Количество блоков
        mergeKernel<<<numBlocks, numThreads>>>(d, temp, w, N);        // Запуск ядра
        cudaMemcpy(d, temp, N * sizeof(int), cudaMemcpyDeviceToDevice);  // Обновляем массив
    }
    cudaDeviceSynchronize();   // Дожидаемся завершения
    end = high_resolution_clock::now(); // Конец замера
    cout << "GPU Merge Sort: " << duration<double, milli>(end - start).count() << " ms\n";

    // --- GPU Quick Sort ---
    cudaMemcpy(d, h.data(), N * sizeof(int), cudaMemcpyHostToDevice); // Сбрасываем данные  
    int segment = 1024;                              // Размер куска для одного потока
    int blocks = (N + segment - 1) / segment;        // Расчет сетки блоков
    start = high_resolution_clock::now();            // Старт времени
    quickSortKernel<<<blocks, 256>>>(d, N, segment); // Запуск параллельной сортировки
    cudaDeviceSynchronize();                         // Синхронизация
    end = high_resolution_clock::now();              // Стоп
    cout << "GPU Quick Sort (simplified): " << duration<double, milli>(end - start).count() << " ms\n";

    // --- GPU Heap Sort ---    
    cudaMemcpy(d, h.data(), N * sizeof(int), cudaMemcpyHostToDevice); // Сбрасываем данные  
    start = high_resolution_clock::now();            // Старт времени
    heapSortKernel<<<blocks, 256>>>(d, N, segment);  // Запуск параллельной сортировки
    cudaDeviceSynchronize();                         // Синхронизация
    end = high_resolution_clock::now();              // Стоп
    cout << "GPU Heap Sort (simplified): " << duration<double, milli>(end - start).count() << " ms\n";

    cudaFree(d);                                    // Освобождаем память на видеокарте
    cudaFree(temp);                                 // Освобождаем временную память
}


int main() {
    vector<int> sizes = {10000, 100000, 1000000}; // Размеры массивов для тестов
    for (int n : sizes) benchmark(n);             // Запуск бенчмарка
    return 0;                                     // Завершение программы
}
