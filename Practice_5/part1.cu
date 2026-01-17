#include <iostream>                 
#include <cuda_runtime.h>           
#include <device_launch_parameters.h>

using namespace std;                

struct Stack {
    int *data;     // Указатель на массив элементов (Глобальная память)
    int *top;      // Указатель на индекс текущей вершины (Глобальная память)
    int capacity;  // Максимально допустимое количество элементов

    // Метод инициализации структуры внутри GPU 
    __device__ void init(int *buffer, int *top_ptr, int size) {
        data = buffer;     // Присваиваем адрес буфера
        top = top_ptr;     // Присваиваем адрес счетчика
        capacity = size;   // Запоминаем предел емкости
    }

    // Параллельная вставка (PUSH)
    __device__ bool push(int value) {
        // atomicAdd возвращает старое значение. Если top был 0, pos станет 0, а top станет 1.
        int pos = atomicAdd(top, 1); 

        if (pos < capacity && pos >= 0) { // Проверка границ (теперь pos начинается с 0)
            data[pos] = value;            // Безопасная запись по уникальному индексу
            return true;                  // Успех
        } else {
            atomicSub(top, 1);            // Откат счетчика, если места нет
            return false;                 // Ошибка: стек полон
        }
    }

    // Параллельное извлечение (POP)
    __device__ bool pop(int *value) {
        // atomicSub возвращает старое значение. Чтобы получить индекс последнего элемента, вычитаем 1.
        int pos = atomicSub(top, 1) - 1; 

        if (pos >= 0) {                   // Если индекс корректен (стек не был пуст)
            *   value = data[pos];            // Считываем данные
            return true;                  // Успех
        } else {
            atomicAdd(top, 1);            // Откат счетчика, если забирать было нечего
            return false;                 // Ошибка: стек пуст
        }
    }
};


// Ядро для заполнения стека
__global__ void testPush(int *buffer, int *top, int capacity, int n) {
    Stack stack;                                     // Создаем локальный экземпляр структуры
    stack.init(buffer, top, capacity);               // Инициализируем указатели внутри потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Вычисляем ID потока
    if (idx < n)                                     // Если поток в рамках задачи
        stack.push(idx * 10);                        // Заталкиваем значение (ID * 10)
}

// Ядро для извлечения из стека
__global__ void testPop(int *buffer, int *top, int capacity, int *results, int n) {
    Stack stack;                                     // Экземпляр структуры
    stack.init(buffer, top, capacity);               // Инициализация
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // ID потока
    if (idx < n) {                                   // Проверка границ
        int val;                                     // Локальная переменная для значения
        if (stack.pop(&val))                          // Пытаемся достать из стека
            results[idx] = val;                      // Записываем в массив итогов
        else
            results[idx] = -1;                       // Пометка, если стек оказался пуст
    }
}


int main() {
    const int N = 100;              // Количество операций (потоков)
    const int CAPACITY = 200;       // Емкость стека на GPU

    int *d_buffer, *d_top, *d_results; // Указатели для видеопамяти

    // 1. Подготовка памяти
    cudaMalloc(&d_buffer, CAPACITY * sizeof(int)); // Выделяем место под данные
    cudaMalloc(&d_top, sizeof(int));               // Выделяем место под переменную вершины
    cudaMalloc(&d_results, N * sizeof(int));       // Место под результаты для проверки

    // Инициализируем top значением 0 (стек пуст, следующая запись в индекс 0)
    int initial_top = 0; 
    cudaMemcpy(d_top, &initial_top, sizeof(int), cudaMemcpyHostToDevice);

    // 2. Параллельное выполнение PUSH
    int threadsPerBlock = 256;                                      // Потоков в блоке
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;       // Расчет блоков
    testPush<<<blocks, threadsPerBlock>>>(d_buffer, d_top, CAPACITY, N); // Запуск
    cudaDeviceSynchronize();                                        // Ожидание завершения

    // Параллельное выполнение POP
    cudaMemset(d_results, -1, N * sizeof(int));                     // Очистка массива итогов
    testPop<<<blocks, threadsPerBlock>>>(d_buffer, d_top, CAPACITY, d_results, N); // Запуск
    cudaDeviceSynchronize();                                        // Ожидание завершения

    // 3. Проверка корректности 
    int *h_results = new int[N];                                    // Массив на стороне CPU
    cudaMemcpy(h_results, d_results, N * sizeof(int), cudaMemcpyDeviceToHost); // Копируем данные

     // Вывод первых 5 извлеченных значений для проверки
    cout << "First 5 values popped from stack (LIFO order): ";
    for (int i = 0; i < 5; i++) cout << h_results[i] << " ";       
    cout << endl;

    // Очистка ресурсов
    delete[] h_results;       // Удаляем массив на CPU
    cudaFree(d_buffer);       // Освобождаем память на GPU
    cudaFree(d_top);          // Освобождаем top
    cudaFree(d_results);      // Освобождаем результаты

    return 0;                 // Завершение программы
}