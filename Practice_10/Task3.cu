#include <iostream>   
#include <vector>     
#include <chrono>     
#include <cuda_runtime.h> 


// Задание 3. Гибридное приложение CPU + GPU с асинхронными копиями
// Макрос для проверки ошибок CUDA: если вызов вернул ошибку, выводим её и завершаем программу
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Кернел (ядро) для выполнения на GPU
__global__ void gpu_kernel(float* data, int n) {
    // Получаем уникальный индекс потока
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float x = data[gid];
        // Искусственная нагрузка: цикл для имитации сложных вычислений
        for (int k = 0; k < 10; ++k) {
            x = x * 1.0001f + 0.0001f;
        }
        data[gid] = x; // Сохраняем результат
    }
}

// Функция для выполнения на CPU
void cpu_work(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        float x = data[i];
        // Аналогичная нагрузка для процессора
        for (int k = 0; k < 10; ++k) {
            x = x * 0.9999f + 0.0002f;
        }
        data[i] = x;
    }
}

int main(int argc, char** argv) {
    // Определяем размер массива (из аргумента или 10 млн элементов)
    int N = (argc > 1) ? std::atoi(argv[1]) : 10000000;

    std::cout << "Hybrid CPU+GPU, N = " << N << std::endl;

    // 1. ОПТИМИЗАЦИЯ: Выделяем Pinned Memory (закрепленную память) на хосте.
    // Это позволяет использовать DMA (прямой доступ к памяти), что ускоряет cudaMemcpyAsync.
    float* h_data = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_data, N * sizeof(float)));

    // Заполняем массив начальными значениями
    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f;
    }

    // Только CPU
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    cpu_work(h_data, N);
    auto t_cpu_end   = std::chrono::high_resolution_clock::now();
    double t_cpu = std::chrono::duration<double>(t_cpu_end - t_cpu_start).count();
    std::cout << "CPU-only time: " << t_cpu << " s\n";

    // Сброс данных для честности теста
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    // Только GPU
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, (N / 2) * sizeof(float))); // Память на GPU для половины данных

    auto t_gpu_start = std::chrono::high_resolution_clock::now();

    // Обычное синхронное копирование: CPU блокируется и ждет
    CUDA_CHECK(cudaMemcpy(d_data, h_data, (N / 2) * sizeof(float), cudaMemcpyHostToDevice));
    
    int n_gpu = N / 2;
    int block = 256;
    int grid  = (n_gpu + block - 1) / block;
    gpu_kernel<<<grid, block>>>(d_data, n_gpu); // Вычисления на GPU
    
    CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения GPU
    // Копирование обратно
    CUDA_CHECK(cudaMemcpy(h_data, d_data, (N / 2) * sizeof(float), cudaMemcpyDeviceToHost));

    auto t_gpu_end   = std::chrono::high_resolution_clock::now();
    double t_gpu = std::chrono::duration<double>(t_gpu_end - t_gpu_start).count();
    std::cout << "GPU-only (sync H2D+kernel+D2H for N/2) time: " << t_gpu << " s\n";

    // Сброс данных
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    // Гибридный
    // Создаем поток CUDA (Stream) для управления очередью команд
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int n_cpu = N / 2;       // Половина для процессора
    n_gpu     = N - n_cpu;   // Остальное для видеокарты

    float* cpu_ptr = h_data;            // Указатель на начало (часть для CPU)
    float* gpu_ptr = h_data + n_cpu;    // Указатель со смещением (часть для GPU)

    auto t_hyb_start = std::chrono::high_resolution_clock::now();

    // 
    
    // Асинхронное копирование: CPU дает команду и СРАЗУ идет дальше
    CUDA_CHECK(cudaMemcpyAsync(d_data, gpu_ptr, n_gpu * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // Запуск ядра в конкретном потоке (не блокирует CPU)
    grid = (n_gpu + block - 1) / block;
    gpu_kernel<<<grid, block, 0, stream>>>(d_data, n_gpu);

    // Асинхронное копирование результата обратно
    CUDA_CHECK(cudaMemcpyAsync(gpu_ptr, d_data, n_gpu * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    // ПОКА GPU ЗАНЯТО ПЕРЕСЫЛКОЙ И РАСЧЕТАМИ, CPU ДЕЛАЕТ СВОЮ ЧАСТЬ!
    cpu_work(cpu_ptr, n_cpu);

    // Теперь нам нужно убедиться, что GPU тоже закончило работу
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto t_hyb_end = std::chrono::high_resolution_clock::now();
    double t_hyb = std::chrono::duration<double>(t_hyb_end - t_hyb_start).count();

    std::cout << "Hybrid (CPU+GPU overlap) time: " << t_hyb << " s\n";

    // Вывод образцов значений для проверки, что расчеты прошли
    std::cout << "Sample values: "
              << "h_data[0] = " << h_data[0]
              << ", h_data[N-1] = " << h_data[N-1] << "\n";

    // Очистка ресурсов
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_data)); // Освобождение pinned-памяти

    return 0; // Завершение программы
}