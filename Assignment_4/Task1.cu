#include <iostream>     
#include <vector>        
#include <random>        
#include <chrono>       
#include <cuda_runtime.h> 


// Задание 1 (25 баллов)
// Реализуйте CUDA-программу для вычисления суммы элементов массива с
// использованием глобальной памяти. Сравните результат и время выполнения с
// последовательной реализацией на CPU для массива размером 100 000 элементов.

// Макрос для проверки ошибок CUDA. Если функция возвращает ошибку, программа выводит сообщение и завершается.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


// Последовательная реализация суммы на CPU
double cpu_sum(const std::vector<float>& v) {
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        s += v[i]; // Просто складываем все элементы в цикле
    }
    return s;
}


// Ядро CUDA: каждый блок считает частичную сумму своего участка данных

__global__ void reduce_sum_kernel(const float* __restrict__ d_in,
                                  float* __restrict__ d_block_sums,
                                  int n) {
    // Объявляем динамическую разделяемую память (shared memory)
    // Она общая для всех потоков внутри ОДНОГО блока и очень быстрая
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x; // ID потока внутри блока
    // Вычисляем глобальный индекс: каждый поток берет два элемента сразу (улучшает нагрузку)
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = 0.0f;
    // Загружаем данные из глобальной памяти в регистр (с проверкой границ массива)
    if (idx < n)         val += d_in[idx];
    if (idx + blockDim.x < n) val += d_in[idx + blockDim.x];

    // Записываем начальную сумму в разделяемую память
    sdata[tid] = val;
    __syncthreads(); // Ждем, пока все потоки блока заполнят sdata

    // Построение "дерева" суммирования внутри блока
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; // Складываем элементы парами
        }
        __syncthreads(); // Синхронизация после каждого шага деления дерева
    }

    // Поток 0 записывает итоговую сумму блока в глобальную память
    if (tid == 0) {
        d_block_sums[blockIdx.x] = sdata[0];
    }
}

// Функция-хост для подготовки данных и запуска GPU
double gpu_sum(const std::vector<float>& h_in, float& elapsed_ms) {
    int n = (int)h_in.size();
    size_t bytes = n * sizeof(float);

    float *d_in = nullptr, *d_block_sums = nullptr;
    // Выделяем память на видеокарте
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    // Копируем данные из ОЗУ в видеопамять
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    int blockSize = 256; // Количество потоков в блоке
    // Так как каждый поток берет по 2 элемента, делим на blockSize * 2
    int gridSize  = (n + blockSize * 2 - 1) / (blockSize * 2); 
    // Выделяем память под частичные суммы от каждого блока
    CUDA_CHECK(cudaMalloc(&d_block_sums, gridSize * sizeof(float)));

    // Создаем события CUDA для замера времени работы ядра
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Запуск ядра. Третий параметр — размер shared memory в байтах.
    reduce_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_in, d_block_sums, n
    );
    
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки запуска
    CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения GPU

    // Копируем частичные суммы обратно на CPU
    std::vector<float> h_block_sums(gridSize);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop)); // Записываем время в мс

    // Очистка ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_block_sums);

    // Финальное суммирование частичных результатов на CPU.
    // Это эффективно, так как элементов в h_block_sums очень мало (gridSize).
    return cpu_sum(h_block_sums);
}

int main() {
    const int N = 100000; // Размер массива
    std::cout << "Task 1: CUDA sum vs CPU, N = " << N << "\n";

    // Инициализация исходного массива случайными числами
    std::vector<float> data(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) data[i] = dist(gen);

    // --- Замер CPU ---
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_res = cpu_sum(data);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // --- Замер GPU ---
    float gpu_ms = 0.0f;
    double gpu_res = gpu_sum(data, gpu_ms);

    // Вывод результатов
    std::cout << "CPU sum  = " << cpu_res << ", time = " << cpu_ms << " ms\n";
    std::cout << "GPU sum  = " << gpu_res << ", kernel time = " << gpu_ms << " ms\n";
    // Проверка точности (может быть небольшая разница из-за порядка сложения float)
    std::cout << "Diff     = " << std::abs(cpu_res - gpu_res) << "\n";

    return 0;
}