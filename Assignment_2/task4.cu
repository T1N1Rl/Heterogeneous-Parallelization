#include <iostream>                   
#include <vector>                      
#include <chrono>                       
#include <cuda_runtime.h>              
#include <device_launch_parameters.h>   
#include <algorithm>                   

using namespace std;                  
using namespace std::chrono;         

__global__ void gpu_merge_kernel(int* d_arr, int* d_temp, int width, int n) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный ID потока

    int start = idx * 2 * width;                 // Каждый поток отвечает за слияние двух подмассивов размером width

    if (start < n) {                            // Проверяем, не вышли ли за границы массива
        int mid = min(start + width, n);        // Определяем середину подмассива
        int end = min(start + 2 * width, n);    // Определяем конец второго подмассива
  
        int i = start;                          // Индекс для левого подмассива
        int j = mid;                            // Индекс для правого подмассива
        int k = start;                          // Индекс для записи результата в временный массив

        while (i < mid && j < end) {            // Основной цикл слияния двух отсортированных подмассивов
            if (d_arr[i] <= d_arr[j]) {         // Если элемент слева меньше или равен
                d_temp[k++] = d_arr[i++];       // Записываем левый, сдвигаем i
            } else {                          
                d_temp[k++] = d_arr[j++];       // Записываем правый, сдвигаем j
            }
        }
 
        while (i < mid) {                       // Если в левой части остались элементы — копируем их
            d_temp[k++] = d_arr[i++];
        }

        while (j < end) {                      // Если в правой части остались элементы — копируем их
            d_temp[k++] = d_arr[j++];
        }
    }
}


void run_test(int N) {
    size_t size = N * sizeof(int);      // Вычисляем размер массива в байтах
    vector<int> h_arr(N);               // Создаём массив на CPU (host)

    for (int i = 0; i < N; i++) {       // Заполняем массив случайными числами
        h_arr[i] = rand() % 10000;      // Генерируем числа от 0 до 9999
    }

    int *d_arr, *d_temp;               // Объявляем указатели на память GPU

    cudaMalloc(&d_arr, size);         // Выделяем память на GPU для основного массива
    cudaMalloc(&d_temp, size);        // Выделяем память на GPU для временного массива

    cudaMemcpy(d_arr, h_arr.data(), size, cudaMemcpyHostToDevice);  // Копируем данные из оперативной памяти (CPU) в видеопамять (GPU)

    auto start_time = high_resolution_clock::now();  // Засекаем время

    for (int width = 1; width < N; width *= 2) {     // Итеративная сортировка слиянием (bottom-up merge sort)
        int num_merges = (N + (2 * width) - 1) / (2 * width);       // Вычисляем количество операций слияния

        int threadsPerBlock = 256;                  // Задаём количество потоков в одном блоке

        int blocksPerGrid = (num_merges + threadsPerBlock - 1) / threadsPerBlock;       // Вычисляем количество блоков

        gpu_merge_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_temp, width, N);  // Запускаем CUDA-ядро

        cudaDeviceSynchronize();        // Дожидаемся завершения выполнения ядра

        cudaMemcpy(d_arr, d_temp, size, cudaMemcpyDeviceToDevice);  // Меняем данные местами: теперь результат слияния в d_temp становится основой
    }

    auto end_time = high_resolution_clock::now();    // Считаем время

    double duration =
        duration_cast<microseconds>(end_time - start_time).count() / 1000.0;  // Вычисляем длительность выполнения в миллисекундах

    cout << "Arrya size: " << N                     
         << " | Time for GPU: " << duration << " ms" << endl;   // Выводим результат измерения времени

    cudaFree(d_arr);    // Освобождаем память на GPU
    cudaFree(d_temp);
}


int main() {
    srand(time(0));   // Инициализируем генератор случайных чисел

    cout << "---  Merge Sort testing on CUDA ---" << endl;

    run_test(10000);   // Тестирование для массива из 10 000 элементов
    run_test(100000);  // Тестирование для массива из 100 000 элементов

    return 0;          // Завершаем программу
}
