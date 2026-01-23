#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// Макрос для автоматической проверки ошибок после каждого вызова API CUDA
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)


// Задание 1: Реализация редукци

// CPU-реализация редукции: последовательная сумма всех элементов.
float cpu_reduce_sum(const std::vector<float>& a) {
    double sum = 0.0;                     // Используем double, для уменьшения накопленной ошибки
    for (size_t i = 0; i < a.size(); ++i) // Линейный проход по массиву
        sum += a[i];                      // Добавляем текущий элемент к сумме
    return static_cast<float>(sum);       // Возвращаем результат в формате float
}

//  Ядро CUDA для параллельной редукции с использованием Shared memory
__global__ void reduce_sum_kernel(const float* __restrict__ d_in,   // Указатель на входные данные в Global Memory
                                  float* __restrict__ d_block_sums, // Массив для записи результатов каждого блока
                                  int n) {                          // Общее количество элементов
    extern __shared__ float sdata[];                                // Объявление динамической Shared memory

    unsigned int tid = threadIdx.x; // Индекс потока внутри текущего блока
    // Вычисляем глобальный индекс: каждый поток берет по 2 элемента для повышения нагрузки (grid-stride)
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = 0.0f;              // Локальная переменная (в регистре) для накопления
    // Загружаем первый элемент, если он в границах массива
    if (idx < n) val += d_in[idx];
    // Загружаем второй элемент со смещением в размер блока, если он в границах
    if (idx + blockDim.x < n) val += d_in[idx + blockDim.x];

    sdata[tid] = val;             // Записываем начальную сумму потока в общую память блока
    __syncthreads();              // Ждем, пока все потоки блока заполнят Разделяемую память

    // Редукция внутри блока: строим дерево суммирования
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { // На каждом шаге делим количество активных потоков пополам
        if (tid < s) {                                      // Если поток в активной половине
            sdata[tid] += sdata[tid + s];                   // Складываем текущее значение с "дальним" соседом
        }
        __syncthreads();         // Синхронизация перед следующим шагом деления дерева
    }

    // После завершения цикла поток 0 содержит сумму всего блока
    if (tid == 0) {
        d_block_sums[blockIdx.x] = sdata[0]; // Записываем итог блока в глобальную память
    }
}

// Хост-функция (CPU) для подготовки и запуска редукции на GPU
float gpu_reduce_sum(const std::vector<float>& h_in, int blockSize, float& elapsed_ms) {
    int n = static_cast<int>(h_in.size());          // Размер входных данных
    size_t bytes = n * sizeof(float);               // Размер в байтах для выделения памяти

    float *d_in = nullptr, *d_block_sums = nullptr; // Указатели для памяти видеокарты

    // Выделяем память на GPU и копируем туда данные с CPU
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // Рассчитываем количество блоков (сетку): n / (blockSize * 2), т.к. поток берет 2 элемента
    int gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
    CUDA_CHECK(cudaMalloc(&d_block_sums, gridSize * sizeof(float))); // Память для промежуточных сумм блоков

    cudaEvent_t start, stop;             // Инструменты для точного замера времени исполнения на GPU
    CUDA_CHECK(cudaEventCreate(&start)); // Создаем событие начала
    CUDA_CHECK(cudaEventCreate(&stop));  // Создаем событие конца
    CUDA_CHECK(cudaEventRecord(start));  // Запускаем "секундомер"

    // Запуск ядра: <<<сетка, блок, объем shared memory в байтах>>>
    reduce_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_block_sums, n);
    
    CUDA_CHECK(cudaGetLastError());      // Проверяем, не было ли ошибок при запуске ядра
    CUDA_CHECK(cudaDeviceSynchronize()); // Ждем завершения всех вычислений на GPU

    std::vector<float> h_block_sums(gridSize); // Резервируем место на CPU для ответов от блоков
    // Копируем частичные суммы из видеопамяти обратно в оперативную память
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Досуммируем результаты блоков на CPU (их обычно немного, это быстро)
    float sum = cpu_reduce_sum(h_block_sums);

    CUDA_CHECK(cudaEventRecord(stop));                          // Останавливаем "секундомер"
    CUDA_CHECK(cudaEventSynchronize(stop));                     // Ждем записи события
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop)); // Вычисляем время в мс

    // Очистка ресурсов
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_block_sums);

    return sum; // Возвращаем финальную сумму
}


// Задание 2: Реализация префиксной суммы

// Алгоритм префиксной суммы на CPU (Exclusive Scan)
void cpu_scan_exclusive(const std::vector<float>& in, std::vector<float>& out) {
    out.resize(in.size()); // Подготавливаем размер выходного массива
    float acc = 0.0f;      // Аккумулятор текущей суммы
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = acc;     // Сначала записываем накопленную сумму (exclusive: текущий элемент не входит)
        acc += in[i];     // Затем прибавляем текущий элемент к аккумулятору
    }
}

// Ядро Blelloch Scan для одного блока GPU
__global__ void scan_blelloch_kernel(const float* __restrict__ d_in, float* __restrict__ d_out, int n) {
    extern __shared__ float temp[]; // Shared memory для построения дерева сумм
    int thid = threadIdx.x;         // ID потока

    int i0 = 2 * thid;     // Индекс первого обрабатываемого элемента
    int i1 = 2 * thid + 1; // Индекс второго обрабатываемого элемента

    // Загружаем данные из Global в Shared memory, заполняя нулями при выходе за границы n
    if (i0 < n) temp[i0] = d_in[i0]; else temp[i0] = 0.0f;
    if (i1 < n) temp[i1] = d_in[i1]; else temp[i1] = 0.0f;

    __syncthreads(); // Ждем загрузки данных всеми потоками

    // 1. Фаза Up-sweep (Reduce): идем снизу вверх по дереву, суммируя значения
    int offset = 1; // Начальное смещение
    for (int d = n >> 1; d > 0; d >>= 1) { // Итерации по уровням дерева (степени двойки)
        __syncthreads();                   // Синхронизация перед каждым уровнем дерева
        if (thid < d) {                    // Если поток активен на этом уровне
            int ai = offset * (2 * thid + 1) - 1; // Индекс левого узла
            int bi = offset * (2 * thid + 2) - 1; // Индекс правого узла
            temp[bi] += temp[ai];                 // Правый узел = сумма левого и правого
        }
        offset <<= 1; // Увеличиваем шаг для следующего уровня
    }

    // Устанавливаем последний элемент в 0 (специфика exclusive scan)
    if (thid == 0) temp[n - 1] = 0.0f;

    // 2. Фаза Down-sweep: спускаемся вниз по дереву, распределяя суммы
    for (int d = 1; d < n; d <<= 1) { // Идем от корня к листьям
        offset >>= 1;                 // Уменьшаем шаг
        __syncthreads();              // Синхронизация перед операциями уровня
        if (thid < d) {               // Если поток активен
            int ai = offset * (2 * thid + 1) - 1; // Левый узел
            int bi = offset * (2 * thid + 2) - 1; // Правый узел
            float t  = temp[ai];      // Сохраняем значение левого узла
            temp[ai] = temp[bi];      // В левый узел записываем текущее значение правого (отца)
            temp[bi] += t;            // В правый узел записываем сумму отца и старого значения левого узла
        }
    }
    __syncthreads(); // Ждем завершения всех обменов

    // Копируем результат из Shared memory обратно в Global memory
    if (i0 < n) d_out[i0] = temp[i0];
    if (i1 < n) d_out[i1] = temp[i1];
}



// Хост-функция для выполнения сканирования (один блок)
void gpu_scan_exclusive(const std::vector<float>& h_in, std::vector<float>& h_out, int blockSize, float& elapsed_ms) {
    int n = static_cast<int>(h_in.size()); // Размер массива
    size_t bytes = n * sizeof(float);      // Объем памяти

    float *d_in = nullptr, *d_out = nullptr; // Указатели GPU
    CUDA_CHECK(cudaMalloc(&d_in, bytes));    // Выделяем входной буфер
    CUDA_CHECK(cudaMalloc(&d_out, bytes));   // Выделяем выходной буфер
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice)); // Копируем данные

    cudaEvent_t start, stop; // Таймеры
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int threads = blockSize; // Количество потоков
    int sharedBytes = n * sizeof(float); // Объем динамической общей памяти

    // Запуск ядра для ОДНОГО блока (ограничение данной реализации: n <= 2 * blockSize)
    scan_blelloch_kernel<<<1, threads, sharedBytes>>>(d_in, d_out, n);

    CUDA_CHECK(cudaDeviceSynchronize()); // Ждем готовности
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop)); // Считаем время

    h_out.resize(n); // Готовим массив на CPU
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost)); // Забираем результат

    cudaFree(d_in); cudaFree(d_out); // Чистим память
    cudaEventDestroy(start); cudaEventDestroy(stop);
}


// Задание 3:  Анализ производительности

int main() {
    cout << "=== Reduction (SUM) CPU vs GPU ===" << endl;

    std::vector<int> sizes = {1024, 1000000, 10000000}; // Размеры тестовых массивов
    int blockSize = 256;                                // Оптимальный размер блока (кратно 32)

    std::mt19937 gen(42);                                   // Инициализация ГПСЧ
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Распределение [0, 1]

    for (int n : sizes) {                                // Цикл по размерам данных
        cout << "\nN = " << n << endl;
        std::vector<float> h_in(n);                      // Создаем вектор на хосте
        for (int i = 0; i < n; ++i) h_in[i] = dist(gen); // Заполняем случайными числами

        // Измеряем CPU
        auto t1 = std::chrono::high_resolution_clock::now();
        float cpu_sum = cpu_reduce_sum(h_in);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // Измеряем GPU
        float gpu_time_ms = 0.0f;
        float gpu_sum = gpu_reduce_sum(h_in, blockSize, gpu_time_ms);

        // Печатаем результаты и разницу (погрешность float)
        cout << "CPU time = " << cpu_time_ms << " ms | GPU time = " << gpu_time_ms << " ms" << endl;
        cout << "Diff = " << fabs(cpu_sum - gpu_sum) << " | Speedup = " << cpu_time_ms / gpu_time_ms << "x" << endl;
    }

    cout << "\n=== Prefix sum (SCAN) CPU vs GPU ===" << endl;

    int n_scan = 1024; // Размер для теста сканирования
    std::vector<float> h_scan_in(n_scan);
    for (int i = 0; i < n_scan; ++i) h_scan_in[i] = dist(gen);

    std::vector<float> cpu_scan; // Результат CPU
    cpu_scan_exclusive(h_scan_in, cpu_scan);

    std::vector<float> gpu_scan; // Результат GPU
    float gpu_scan_time_ms = 0.0f;
    gpu_scan_exclusive(h_scan_in, gpu_scan, 512, gpu_scan_time_ms);

    // Сравниваем первые 8 элементов для визуальной проверки
    cout << "GPU scan time = " << gpu_scan_time_ms << " ms" << endl;
    cout << "\nFirst 8 elements (input / CPU scan / GPU scan):" << endl;
    for (int i = 0; i < 8; ++i) {
        cout << i << ": " << h_scan_in[i] << " / " << cpu_scan[i] << " / " << gpu_scan[i] << endl;
    }

    return 0; // Завершение программы
}