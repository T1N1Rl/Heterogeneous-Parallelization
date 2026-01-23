#include <iostream>  
#include <vector>     
#include <chrono>    
#include <cuda_runtime.h> 

// Задание 3 (25 баллов)
// Реализуйте гибридную программу, в которой обработка массива выполняется
// параллельно на CPU и GPU. Первую часть массива обработайте на CPU, вторую — на
// GPU. Сравните время выполнения CPU-, GPU- и гибридной реализаций.

// Макрос для автоматической проверки ошибок CUDA. Если вызов функции неудачен, выводит текст ошибки.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Ядро (kernel) для выполнения на GPU
__global__ void gpu_kernel(float* data, int n) {
    // Вычисляем глобальный индекс потока
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // Проверка на выход за границы массива
    if (gid < n) {
        float x = data[gid];
        // Имитация тяжелых вычислений (цикл из 10 операций)
        for (int k = 0; k < 10; ++k) {
            x = x * 1.0001f + 0.0001f;
        }
        data[gid] = x; // Сохраняем результат обратно в глобальную память
    }
}

// Функция для выполнения на CPU
void cpu_kernel(float* data, int n) {
    // Обычный последовательный цикл
    for (int i = 0; i < n; ++i) {
        float x = data[i];
        // Имитация нагрузки (чуть иная формула для отличия результатов)
        for (int k = 0; k < 10; ++k) {
            x = x * 0.9999f + 0.0002f;
        }
        data[i] = x;
    }
}

int main(int argc, char** argv) {
    // Определяем размер массива: из аргумента командной строки или 10 млн по умолчанию
    int N = (argc > 1) ? std::atoi(argv[1]) : 10000000;
    std::cout << "Task 3: Hybrid CPU+GPU, N = " << N << "\n";

    // Создаем три вектора для трех разных тестов
    std::vector<float> data_cpu(N, 1.0f); // Только для CPU
    std::vector<float> data_gpu(N, 1.0f); // Только для GPU
    std::vector<float> data_hyb(N, 1.0f); // Для гибридной схемы

    // Только CPU
    auto t_cpu_start = std::chrono::high_resolution_clock::now(); // Засекаем старт
    cpu_kernel(data_cpu.data(), N); // Выполняем расчеты
    auto t_cpu_end = std::chrono::high_resolution_clock::now();   // Засекаем конец
    double t_cpu = std::chrono::duration<double>(t_cpu_end - t_cpu_start).count();
    std::cout << "CPU-only time: " << t_cpu << " s\n";

    // Только GPU
    float *d_data = nullptr;
    // Выделяем память на видеокарте под весь массив
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    // Копируем данные из оперативной памяти в видеопамять
    CUDA_CHECK(cudaMemcpy(d_data, data_gpu.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto t_gpu_start = std::chrono::high_resolution_clock::now();
    int block = 256; // 256 потоков в блоке
    int grid  = (N + block - 1) / block; // Вычисляем количество блоков
    gpu_kernel<<<grid, block>>>(d_data, N); // Запускаем вычисления на GPU
    
    CUDA_CHECK(cudaDeviceSynchronize()); // Ждем, пока GPU закончит работу
    // Копируем результат обратно в ОЗУ
    CUDA_CHECK(cudaMemcpy(data_gpu.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    double t_gpu = std::chrono::duration<double>(t_gpu_end - t_gpu_start).count();
    std::cout << "GPU-only (copy+kernel+copy) time: " << t_gpu << " s\n";

    // Гибридный
    std::fill(data_hyb.begin(), data_hyb.end(), 1.0f); // Сбрасываем данные

    int n_cpu = N / 2;       // Первая половина массива для CPU
    int n_gpu = N - n_cpu;   // Вторая половина массива для GPU

    float* h_cpu = data_hyb.data();            // Указатель на начало "процессорной" части
    float* h_gpu = data_hyb.data() + n_cpu;    // Указатель на начало "видеокарточной" части

    float* d_seg = nullptr;
    // Выделяем на видеокарте память только под половину данных
    CUDA_CHECK(cudaMalloc(&d_seg, n_gpu * sizeof(float)));

    auto t_hyb_start = std::chrono::high_resolution_clock::now();

    // Отправляем данные второй половины на GPU
    CUDA_CHECK(cudaMemcpy(d_seg, h_gpu, n_gpu * sizeof(float), cudaMemcpyHostToDevice));
    int grid2 = (n_gpu + block - 1) / block;
    // Запускаем GPU ядро. Оно работает асинхронно — управление сразу возвращается CPU!
    gpu_kernel<<<grid2, block>>>(d_seg, n_gpu);

    // В ЭТО ЖЕ ВРЕМЯ CPU обрабатывает первую половину
    cpu_kernel(h_cpu, n_cpu);

    // Дожидаемся окончания работы GPU (синхронизация)
    CUDA_CHECK(cudaDeviceSynchronize());
    // Возвращаем результат GPU-части в общий массив
    CUDA_CHECK(cudaMemcpy(h_gpu, d_seg, n_gpu * sizeof(float), cudaMemcpyDeviceToHost));

    auto t_hyb_end = std::chrono::high_resolution_clock::now();
    double t_hyb = std::chrono::duration<double>(t_hyb_end - t_hyb_start).count();

    std::cout << "Hybrid time (CPU+GPU overlap): " << t_hyb << " s\n";

    // Вывод образцовых значений для проверки корректности
    std::cout << "Sample values:\n";
    std::cout << "  CPU    data_cpu[0] = " << data_cpu[0] << "\n";
    std::cout << "  GPU    data_gpu[0] = " << data_gpu[0] << "\n";
    std::cout << "  Hybrid data_hyb[0] = " << data_hyb[0] << "\n";

    // Очистка памяти
    cudaFree(d_data);
    cudaFree(d_seg);
    return 0;
}