#include <iostream>   
#include <vector>   
#include <random>   
#include <chrono>     
#include <cmath>    
#include <cuda_runtime.h> 
#include <omp.h>       

using namespace std;

// Макрос для проверки ошибок CUDA. Если функция вернет ошибку, программа выведет сообщение и завершится.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


const int N = 1'000'000;       // Общий размер массива (миллион элементов)
const float MULTIPLIER = 2.0f; // Число, на которое будем умножать элементы

// GPU ядро
// __global__ указывает, что функция вызывается с CPU, а исполняется на GPU
__global__ void multiply_kernel(float* d_data, int n, float multiplier) {
    // Вычисляем глобальный индекс потока в сетке CUDA
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверка, чтобы индекс не вышел за пределы массива
    if (idx < n) {
        d_data[idx] *= multiplier; // Сама операция умножения в памяти видеокарты
    }
}

// CPU: использует OpenMP для распараллеливания на ядра процессора
void process_cpu_omp(float* data, int n, float multiplier) {
    // Директива OpenMP: разделить итерации цикла между всеми доступными ядрами CPU
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        data[i] *= multiplier; // Операция умножения в оперативной памяти
    }
}

// GPU: управляет памятью и запуском ядра
void process_gpu_segment(float* h_segment, int n, float multiplier, float& elapsed_ms) {
    size_t bytes = n * sizeof(float); // Рассчитываем размер памяти в байтах
    float* d_data = nullptr;          // Указатель для памяти на устройстве (видеокарте)

    // Выделяем память в VRAM видеокарты
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Засекаем время начала (включая пересылки данных)
    auto t1 = std::chrono::high_resolution_clock::now();

    // Копируем данные из оперативной памяти (Host) в видеопамять (Device) — H2D
    CUDA_CHECK(cudaMemcpy(d_data, h_segment, bytes, cudaMemcpyHostToDevice));

    // Настройка конфигурации запуска: сколько потоков в блоке и сколько блоков в сетке
    int blockSize = 256; // Стандартный размер блока потоков
    int gridSize  = (n + blockSize - 1) / blockSize; // Расчет количества блоков, чтобы покрыть N элементов
    
    // Запуск ядра на GPU (асинхронно)
    multiply_kernel<<<gridSize, blockSize>>>(d_data, n, multiplier);
    
    // Проверка, не возникло ли ошибок при запуске ядра
    CUDA_CHECK(cudaGetLastError());
    // Ожидание завершения работы GPU (синхронизация)
    CUDA_CHECK(cudaDeviceSynchronize());

    // Копируем результат обратно из видеопамяти в оперативную — D2H
    CUDA_CHECK(cudaMemcpy(h_segment, d_data, bytes, cudaMemcpyDeviceToHost));

    auto t2 = std::chrono::high_resolution_clock::now();
    // Считаем время выполнения блока в миллисекундах
    elapsed_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Освобождаем выделенную память на видеокарте
    CUDA_CHECK(cudaFree(d_data));
}

// Функция для проверки точности расчетов (сравнение результатов)
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY; // Если размеры разные, возвращаем бесконечность
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]); // Вычисляем разницу по модулю между элементами
        if (d > max_diff) max_diff = d;  // Ищем максимальное расхождение
    }
    return max_diff;
}

int main() {
    std::cout << "Hybrid CPU+GPU lab (N = " << N << ")\n";

    // 1. Исходные данные
    std::vector<float> h_src(N); // Создаем массив на N элементов
    std::mt19937 gen(42);        // Инициализируем генератор случайных чисел зерном 42
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Распределение от 0 до 1
    for (int i = 0; i < N; ++i) {
        h_src[i] = dist(gen); // Заполняем массив случайными числами
    }

    // Создаем копии массива для разных тестов
    std::vector<float> h_cpu   = h_src; // Для чистого OpenMP
    std::vector<float> h_gpu   = h_src; // Для чистого CUDA
    std::vector<float> h_hyb   = h_src; // Для гибридного режима

    // Задание 1: Толко CPU
    std::cout << "\n=== CPU-only (OpenMP) ===\n";
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    process_cpu_omp(h_cpu.data(), N, MULTIPLIER); // Запуск многопоточной обработки на CPU
    auto t_cpu_end   = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count();
    std::cout << "CPU (OpenMP) time: " << cpu_time_ms << " ms\n";

    // Задание 2: Только GPU (CUDA)
    std::cout << "\n=== GPU-only (CUDA) ===\n";
    float gpu_time_ms = 0.0f;
    // Весь массив передается на видеокарту, обрабатывается и возвращается
    process_gpu_segment(h_gpu.data(), N, MULTIPLIER, gpu_time_ms);

    // Сравнение результатов CPU и GPU для проверки корректности
    float diff_cpu_gpu = max_abs_diff(h_cpu, h_gpu);
    std::cout << "GPU total time (H2D + kernel + D2H): " << gpu_time_ms << " ms\n";
    std::cout << "Max |CPU - GPU| diff = " << diff_cpu_gpu << "\n";
    std::cout << "Speedup GPU vs CPU approx " << cpu_time_ms / gpu_time_ms << "x\n";

    // Задание 3: Гибридный 
    std::cout << "\n=== HYBRID (CPU + GPU) ===\n";

    int n_cpu = N / 2;       // Половина элементов для процессора
    int n_gpu = N - n_cpu;   // Вторая половина для видеокарты

    float* cpu_ptr = h_hyb.data();          // Указатель на начало первой половины
    float* gpu_ptr = h_hyb.data() + n_cpu;  // Указатель на начало второй половины

    float gpu_part_time_ms = 0.0f;
    auto t_hyb_start = std::chrono::high_resolution_clock::now();

    // Директива для одновременного выполнения разных кусков кода
    #pragma omp parallel sections
    {
        // Первая секция: выполняется на одном потоке CPU (который внутри запустит еще потоки через omp for)
        #pragma omp section
        {
            process_cpu_omp(cpu_ptr, n_cpu, MULTIPLIER);
        }

        // Вторая секция: выполняется параллельно первой, отправляет задачу на видеокарту
        #pragma omp section
        {
            process_gpu_segment(gpu_ptr, n_gpu, MULTIPLIER, gpu_part_time_ms);
        }
    }

    auto t_hyb_end = std::chrono::high_resolution_clock::now();
    double hyb_time_ms = std::chrono::duration<double, std::milli>(t_hyb_end - t_hyb_start).count();

    std::cout << "Hybrid time (CPU+GPU overlap): " << hyb_time_ms << " ms\n";

    // Проверка: гибридный результат должен совпадать с эталонным CPU-результатом
    float diff_cpu_hyb = max_abs_diff(h_cpu, h_hyb);
    std::cout << "Max |CPU - HYB| diff = " << diff_cpu_hyb << "\n";

    // Задание 4: Анализ
    std::cout << "Speedup HYB vs CPU approx " << cpu_time_ms / hyb_time_ms << "x\n";
    std::cout << "Speedup HYB vs GPU approx " << gpu_time_ms / hyb_time_ms << "x\n";

    // Вывод первых 8 элементов для визуального подтверждения правильности
    std::cout << "\nFirst 8 elements (source / CPU / GPU / HYB):\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << i << ": " << h_src[i] << " / " << h_cpu[i] << " / " << h_gpu[i] << " / " << h_hyb[i] << "\n";
    }

    std::cout << "\nDone.\n";
    return 0; // Завершение программы
}