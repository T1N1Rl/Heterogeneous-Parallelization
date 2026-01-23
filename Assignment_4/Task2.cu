#include <iostream>     
#include <vector>       
#include <random>       
#include <chrono>      
#include <cuda_runtime.h> 

//Задание 2 (25 баллов)
// Реализуйте CUDA-программу для вычисления префиксной суммы (сканирования)
// массива с использованием разделяемой памяти. Сравните время выполнения с
// последовательной реализацией на CPU для массива размером 1 000 000 элементов.

// Макрос для автоматической проверки ошибок CUDA функций
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Последовательная версия: проходим циклом и накапливаем сумму
void cpu_scan_exclusive(const std::vector<float>& in, std::vector<float>& out) {
    int n = (int)in.size();
    out.resize(n);
    float acc = 0.0f; // Аккумулятор суммы
    for (int i = 0; i < n; ++i) {
        out[i] = acc; // Сначала записываем текущую сумму
        acc += in[i]; // Затем прибавляем текущий элемент (эксклюзивность)
    }
}


// Ядро для параллельного сканирования внутри одного блока потоков

__global__ void scan_block_kernel(const float* __restrict__ d_in,
                                  float* __restrict__ d_out,
                                  float* __restrict__ d_block_sums,
                                  int n) {
    // Выделяем разделяемую память (shared memory) для блока
    extern __shared__ float temp[];
    int tid = threadIdx.x; // Индекс потока в блоке
    int gid = blockIdx.x * blockDim.x * 2 + tid; // Глобальный индекс

    // Загружаем по 2 элемента на каждый поток для эффективности (Coalesced access)
    int i0 = gid;
    int i1 = gid + blockDim.x;

    float v0 = (i0 < n) ? d_in[i0] : 0.0f;
    float v1 = (i1 < n) ? d_in[i1] : 0.0f;

    temp[2 * tid]     = v0;
    temp[2 * tid + 1] = v1;

    int offset = 1;
    int N = 2 * blockDim.x; // Общее число элементов, обрабатываемых блоком

    // ФАЗА 1: Up-sweep (Reduce). Строим дерево сумм снизу вверх.
    for (int d = N >> 1; d > 0; d >>= 1) {
        __syncthreads(); // Синхронизируем потоки перед шагом
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Сохраняем общую сумму блока в отдельный массив (нужно для объединения блоков)
    if (tid == 0) {
        d_block_sums[blockIdx.x] = temp[N - 1];
        temp[N - 1] = 0.0f; // Устанавливаем 0 для начала обратного хода (exclusive)
    }

    // ФАЗА 2: Down-sweep. Спускаемся по дереву вниз для формирования префиксных сумм.
    for (int d = 1; d < N; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];   // Левому потомку отдаем значение родителя
            temp[bi] += t;         // Правому — сумму родителя и левого потомка
        }
    }
    __syncthreads();

    // Выгружаем результат из shared memory обратно в глобальную видеопамять
    if (i0 < n) d_out[i0] = temp[2 * tid];
    if (i1 < n) d_out[i1] = temp[2 * tid + 1];
}

// Дополнительное ядро: добавляет к каждому элементу блока общую сумму всех предыдущих блоков
__global__ void add_block_offsets_kernel(float* d_data,
                                         const float* __restrict__ d_block_offsets,
                                         int n) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gid0 = bid * blockDim.x * 2 + tid;
    int gid1 = gid0 + blockDim.x;

    float offset = d_block_offsets[bid]; // Сумма, накопленная предыдущими блоками

    if (gid0 < n) d_data[gid0] += offset;
    if (gid1 < n) d_data[gid1] += offset;
}

// Основная функция запуска сканирования на GPU
void gpu_scan_exclusive(const std::vector<float>& h_in,
                        std::vector<float>& h_out,
                        float& elapsed_ms) {
    int n = (int)h_in.size();
    size_t bytes = n * sizeof(float);

    float *d_in = nullptr, *d_out = nullptr;
    // Выделение памяти на видеокарте
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    int blockSize = 512; // 512 потоков = 1024 элемента на блок
    int elementsPerBlock = 2 * blockSize;
    int gridSize = (n + elementsPerBlock - 1) / elementsPerBlock;

    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, gridSize * sizeof(float)));

    // Замер времени через события CUDA
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // ШАГ 1: Локальное сканирование внутри каждого блока
    size_t shmem = elementsPerBlock * sizeof(float);
    scan_block_kernel<<<gridSize, blockSize, shmem>>>(
        d_in, d_out, d_block_sums, n
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ШАГ 2: Сканирование самих сумм блоков (на CPU, так как их мало)
    std::vector<float> h_block_sums(gridSize);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> h_block_offsets(gridSize);
    cpu_scan_exclusive(h_block_sums, h_block_offsets); // Вычисляем смещения для блоков

    float* d_block_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_offsets, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_block_offsets, h_block_offsets.data(),
                          gridSize * sizeof(float), cudaMemcpyHostToDevice));

    // ШАГ 3: Финальная коррекция — добавляем смещения к локальным суммам
    add_block_offsets_kernel<<<gridSize, blockSize>>>(
        d_out, d_block_offsets, n
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Копируем итоговый результат обратно в ОЗУ
    h_out.resize(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_block_sums);
    cudaFree(d_block_offsets);
}

int main() {
    const int N = 1000000; // 1000000 элементов
    std::cout << "Task 2: prefix sum (exclusive), N = " << N << "\n";

    // Генерация входных данных
    std::vector<float> in(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) in[i] = dist(gen);

    std::vector<float> cpu_out, gpu_out;

    // Расчет на CPU
    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_scan_exclusive(in, cpu_out);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Расчет на GPU
    float gpu_ms = 0.0f;
    gpu_scan_exclusive(in, gpu_out, gpu_ms);

    // Проверка точности (сравнение результатов)
    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float d = std::fabs(cpu_out[i] - gpu_out[i]);
        if (d > max_diff) max_diff = d;
    }

    std::cout << "CPU time = " << cpu_ms << " ms\n";
    std::cout << "GPU time (kernel part) = " << gpu_ms << " ms\n";
    std::cout << "Max diff = " << max_diff << "\n";

    return 0;
}