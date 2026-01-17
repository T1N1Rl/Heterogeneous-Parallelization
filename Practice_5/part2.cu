#include <iostream>                 
#include <cuda_runtime.h>           
#include <device_launch_parameters.h>

using namespace std;

struct Queue {
    int *data;      // Указатель на массив элементов в Глобальной памяти
    int *head;      // Указатель на индекс начала очереди
    int *tail;      // Указатель на индекс конца очереди
    int capacity;   // Максимальный размер очереди

    // Метод инициализации указателей Queue
    __device__ void init(int *buffer, int *h_ptr, int *t_ptr, int size) {
        data = buffer;    // Связываем указатель data с выделенным буфером
        head = h_ptr;     // Привязываем указатель на голову к адресу в глобальной памяти
        tail = t_ptr;     // Привязываем указатель на хвост к адресу в глобальной памяти
        capacity = size;  // Устанавливаем лимит элементов
    }

    // Безопасное добавление в конец Enqueue
    __device__ bool enqueue(int value) {
        // Резервируем место в хвосте
        int pos = atomicAdd(tail, 1);

        if (pos < capacity) {  // Проверяем, не переполнен ли буфер
            data[pos] = value; // Запись данных
            return true;       // Возвращаем успех
        }
        // Если очередь переполнена, увеличивать дальше нельзя (простая реализация без кольца)
        return false;
    }

    // Безопасное удаление из начала Dequeue
    __device__ bool dequeue(int &value) {
        // Резервируем индекс для чтения из головы очереди
        int pos = atomicAdd(head, 1);

        // Проверяем, не обогнала ли голова хвост (есть ли элементы)
        // Используем чтение хвоста в данный момент
        if (pos < atomicAdd(tail, 0)) { 
            value = data[pos]; // Чтение данных из буфера
            return true;       // Успех извлечения
        }
        return false;          // Ошибка: данных нет
    }
};



// Ядра CUDA

__global__ void testQueue(int *buffer, int *head, int *tail, int capacity, int *results, int n) {
    Queue queue;                                 // Создаем локальный (в регистрах потока) экземпляр структуры Queue
    queue.init(buffer, head, tail, capacity);    // Инициализируем её общими адресами

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Вычисляем уникальный ID потока
    if (idx < n) {                                    // Проверка границ массива
        // Сначала все потоки добавляют элементы
        queue.enqueue(idx * 5);
        
        // Синхронизация, чтобы было что извлекать
        __syncthreads();

        int val;
        // Затем пытаются извлечь
        if (queue.dequeue(val)) {
            results[idx] = val;     // Если извлекли — пишем в массив результатов
        } else {           
            results[idx] = -1;      // Если очередь пуста — пишем маркер ошибки
        }
    }
}


int main() {
    const int N = 100;             // Количество тестовых операций (потоков)
    const int CAPACITY = 1000;     // Размер выделенного буфера очереди

    int *d_buffer, *d_head, *d_tail, *d_results;    // Указатели для памяти на видеокарте
    // Выделение памяти в VRAM (Глобальная память)
    cudaMalloc(&d_buffer, CAPACITY * sizeof(int));  // Буфер данных
    cudaMalloc(&d_head, sizeof(int));               // Переменная индекса головы
    cudaMalloc(&d_tail, sizeof(int));               // Переменная индекса хвоста
    cudaMalloc(&d_results, N * sizeof(int));        // Буфер для проверки итогов на CPU

    // 1. Инициализация указателей
    int zero = 0;
    cudaMemcpy(d_head, &zero, sizeof(int), cudaMemcpyHostToDevice);  // Копируем 0 в d_head
    cudaMemcpy(d_tail, &zero, sizeof(int), cudaMemcpyHostToDevice);  // Копируем 0 в d_tail

    // 2. Запуск очереди
    testQueue<<<1, N>>>(d_buffer, d_head, d_tail, CAPACITY, d_results, N);  // 1 блок, N потоков
    cudaDeviceSynchronize();  // Синхронизация CPU и GPU: ждем завершения работы видеокарты

    // Копирование результатов обратно на хост (процессор)
    int *h_results = new int[N];                                                   // Массив в оперативной памяти
    cudaMemcpy(h_results, d_results, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Вывод первых 5 извлеченных значений для проверки
    cout << "First 5 values dequeued from queue (FIFO order): ";
    for (int i = 0; i < 5; i++) cout << h_results[i] << " ";
    cout << endl;

    // Очистка ресурсов
    cudaFree(d_buffer); cudaFree(d_head); cudaFree(d_tail); cudaFree(d_results);
    delete[] h_results;  // Удаление массива на CPU

    return 0;            // Завершение программы
}