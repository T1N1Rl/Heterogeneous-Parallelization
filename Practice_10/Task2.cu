#include <iostream>      
#include <vector>        
#include <cuda_runtime.h> 


// Задание 2. Оптимизация доступа к памяти на GPU (CUDA)
// Макрос для автоматической проверки ошибок CUDA после выполнения функций
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// 1а. Эффективный (коалесцированный) доступ
// Потоки одного варпа (32 потока) читают соседние ячейки памяти одновременно

__global__ void kernel_coalesced(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n) {
    // Вычисляем глобальный индекс текущего потока
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Если индекс внутри границ массива
    if (gid < n) {
        float x = in[gid];     // Чтение идет "подряд": поток 0 берет элемент 0, поток 1 — элемент 1
        out[gid] = x * 2.0f;   // Запись также идет последовательно
    }
}

// 1б. Неэффективный (некоалесцированный) доступ
// Потоки одного варпа обращаются к памяти с большим шагом (stride)
__global__ void kernel_noncoalesced(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n,
                                    int stride) {
    // Вычисляем глобальный индекс потока
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // Создаем "скачущий" индекс: потоки лезут в разные сегменты памяти
    int idx = (gid * stride) % n; 
    
    float x = in[idx];         // Видеокарта вынуждена делать много отдельных запросов к памяти
    out[idx] = x * 2.0f;
}

// 3. Оптимизация с использованием разделяемой (Shared) памяти
// Сначала копируем данные в быструю память внутри блока, потом работаем с ними

__global__ void kernel_shared(const float* __restrict__ in,
                              float* __restrict__ out,
                              int n) {
    // Объявляем массив в разделяемой памяти (tile), размер задается при запуске ядра
    extern __shared__ float tile[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный ID
    int tid = threadIdx.x;                           // Локальный ID внутри блока

    if (gid < n) {
        // Коалесцированное (быстрое) чтение из глобальной памяти в shared память
        tile[tid] = in[gid]; 
    }
    // Барьерная синхронизация: ждем, пока все потоки блока закончат запись в tile
    __syncthreads();

    if (gid < n) {
        // Чтение из сверхбыстрой Shared памяти
        float x = tile[tid];
        out[gid] = x * 2.0f;
    }
}

// Функция-обертка для замера времени коалесцированного ядра
float run_kernel_coalesced(const float* d_in, float* d_out, int n) {
    int block = 256; // Количество потоков в блоке
    int grid = (n + block - 1) / block; // Количество блоков
    
    cudaEvent_t start, stop; // События CUDA для точного замера времени на GPU
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start)); // Фиксируем старт
    kernel_coalesced<<<grid, block>>>(d_in, d_out, n); // Запуск ядра
    CUDA_CHECK(cudaEventRecord(stop));  // Фиксируем стоп
    
    CUDA_CHECK(cudaEventSynchronize(stop));             // Ждем завершения записи стоп-события
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Считаем разницу в мс
    
    cudaEventDestroy(start);                            // Очистка ресурсов событий
    cudaEventDestroy(stop);
    return ms;
}

// Функция-обертка для замера времени ядра со "скачками" по памяти
float run_kernel_noncoalesced(const float* d_in, float* d_out, int n, int stride) {
    int block = 256;
    int grid = (n + block - 1) / block;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    kernel_noncoalesced<<<grid, block>>>(d_in, d_out, n, stride);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Функция-обертка для замера времени ядра с Shared памятью
float run_kernel_shared(const float* d_in, float* d_out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    size_t shmem = block * sizeof(float); // Объем Shared памяти на один блок
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    // Третий параметр в <<< >>> — размер динамической Shared памяти
    kernel_shared<<<grid, block, shmem>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main(int argc, char** argv) {
    // Размер массива из аргументов или 1 млн
    int N = (argc > 1) ? std::atoi(argv[1]) : 1000000;
    // Шаг (stride) специально сделан кратным 32 (размеру варпа), что ломает кэширование
    int stride = 32 * 16; 

    std::cout << "CUDA memory access patterns, N = " << N
              << ", stride = " << stride << std::endl;

    // Резервируем память на CPU
    std::vector<float> h_in(N, 1.0f), h_out(N, 0.0f);

    float *d_in = nullptr, *d_out = nullptr;
    // Выделяем память на GPU
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    // Копируем исходные данные на GPU
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Запускаем три разных теста
    float t_coal = run_kernel_coalesced(d_in, d_out, N);
    float t_non  = run_kernel_noncoalesced(d_in, d_out, N, stride);
    float t_sh   = run_kernel_shared(d_in, d_out, N);

    // Вывод результатов замеров
    std::cout << "Coalesced kernel:     " << t_coal << " ms\n";
    std::cout << "Non-coalesced kernel:" << t_non  << " ms\n";
    std::cout << "Shared-optimized:     " << t_sh   << " ms\n";

    // Копируем один результат назад для проверки, что программа сработала
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Sample output[0] = " << h_out[0] << std::endl;

    // Освобождение памяти
    cudaFree(d_in);
    cudaFree(d_out);
    return 0; // Завершение программы
}