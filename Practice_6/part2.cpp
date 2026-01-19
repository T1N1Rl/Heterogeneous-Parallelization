#include <iostream>             
#include <vector>               
#include <fstream>              
#include <chrono>               
#include <cmath>                
#include <CL/cl.h>              

using namespace std;             

// Чтение OpenCL-ядра из файла
string readKernel(const string& fileName) {
    ifstream file(fileName);     // Открываем файл с исходником ядра
    return string((istreambuf_iterator<char>(file)), // Читаем файл целиком в std::string
                  istreambuf_iterator<char>());
}

// Простая проверка ошибок OpenCL
void checkError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {     // Если код ошибки не равен CL_SUCCESS
        cerr << "OpenCL error (" << err << "): " << msg << endl; // Логируем ошибку
        exit(1);                 // Завершаем программу с ошибкой
    }
}

int main() {
    // Параметры матриц (A[NxM] * B[MxK] = C[NxK])
    const int N = 512, M = 512, K = 512; // Задаём размерности матриц

    // Подготовка данных на CPU
    vector<float> m_A(N * M, 1.5f);      // Матрица A на хосте, заполнена 1.5
    vector<float> m_B(M * K, 0.5f);      // Матрица B на хосте, заполнена 0.5

    // Результаты
    vector<float> m_C_cpu(N * K, 0.0f);  // Результирующая матрица C для CPU-версии
    vector<float> m_C_gpu(N * K, 0.0f);  // Результирующая матрица C для GPU (OpenCL)

    // CPU
    auto cpu_start = chrono::high_resolution_clock::now(); // Фиксируем время начала CPU-вычислений

    for (int row = 0; row < N; ++row) {          // Проходим по строкам результирующей матрицы
        for (int col = 0; col < K; ++col) {      // Проходим по столбцам результирующей матрицы
            float sum = 0.0f;                    // Аккумулятор суммы для элемента C[row, col]
            for (int i = 0; i < M; ++i) {        // Скалярное произведение строки A и столбца B
                sum += m_A[row * M + i] *        // Берём элемент A[row, i]
                       m_B[i * K + col];         // Умножаем на элемент B[i, col]
            }
            m_C_cpu[row * K + col] = sum;        // Записываем результат в C_cpu[row, col]
        }
    }

    auto cpu_end = chrono::high_resolution_clock::now(); // Фиксируем время окончания CPU-вычислений
    double cpu_ms = chrono::duration<double, milli>(cpu_end - cpu_start).count(); // Считаем время CPU в миллисекундах

    cout << "Matrix size: " << N << "x" << M << " * " << M << "x" << K << endl; // Логируем размер матриц
    cout << "CPU time: " << cpu_ms << " ms" << endl;                            // Логируем время выполнения на CPU

    // GPU

    cl_int err;                                // Переменная для хранения кодов ошибок OpenCL

    // Платформа и устройство
    cl_platform_id platform;                   // Идентификатор платформы OpenCL
    err = clGetPlatformIDs(1, &platform, nullptr); // Получаем первую доступную платформу
    checkError(err, "clGetPlatformIDs");       // Проверяем, что вызов успешен

    cl_device_id device;                       // Идентификатор устройства (GPU)
    // Если нужно замерять на CPU, заменить CL_DEVICE_TYPE_GPU на CL_DEVICE_TYPE_CPU (при наличии CPU-устройства OpenCL)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr); // Получаем одно GPU-устройство
    checkError(err, "clGetDeviceIDs");         // Проверяем корректность

    // Контекст и очередь
    cl_context context = clCreateContext(nullptr, // Параметры по умолчанию (без доп. callback’ов)
                                         1,       // Количество устройств в контексте
                                         &device, // Указатель на устройство
                                         nullptr, // Callback об ошибках (не используем)
                                         nullptr, // Дополнительные данные для callback (не используем)
                                         &err);   // Код ошибки
    checkError(err, "clCreateContext");        // Проверка создания контекста

    cl_command_queue queue = clCreateCommandQueue(context, // Контекст
                                                  device,  // Устройство
                                                  0,       // Флаги (без профайлинга)
                                                  &err);   // Код ошибки
    checkError(err, "clCreateCommandQueue");   // Проверка создания очереди команд

    // Сборка программы из файла
    string sourceStr = readKernel("kernels.cl"); // Читаем исходный текст OpenCL-ядра из файла
    const char* source = sourceStr.c_str();      // Получаем C-строку из std::string
    size_t sourceSize = sourceStr.size();        // Размер исходника в байтах

    cl_program program = clCreateProgramWithSource(context, // Контекст
                                                   1,       // Количество строк исходника
                                                   &source, // Указатель на строку с исходником
                                                   &sourceSize, // Указатель на размер строки
                                                   &err);   // Код ошибки
    checkError(err, "clCreateProgramWithSource"); // Проверяем создание программы

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr); // Компилируем программу под выбранное устройство
    if (err != CL_SUCCESS) {                     // Если сборка неуспешна
        size_t log_size = 0;                     // Переменная для размера лога сборки
        clGetProgramBuildInfo(program, device,
                              CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size); // Узнаём размер build-лога
        vector<char> log(log_size);              // Выделяем буфер под лог
        clGetProgramBuildInfo(program, device,
                              CL_PROGRAM_BUILD_LOG,
                              log_size, log.data(), nullptr); // Читаем сам лог
        cerr << "Build log:\n" << log.data() << endl; // Печатаем лог компиляции
        checkError(err, "clBuildProgram");       // Отдаём ошибку дальше
    }

    cl_kernel kernelMul = clCreateKernel(program, // Программа, где лежит ядро
                                         "matrix_mul", // Имя ядра в kernels.cl
                                         &err);   // Код ошибки
    checkError(err, "clCreateKernel");           // Проверяем создание ядра

    // Буферы на устройстве
    cl_mem d_matA = clCreateBuffer(context,               // Контекст
                                   CL_MEM_READ_ONLY |     // Буфер только для чтения в ядре
                                   CL_MEM_COPY_HOST_PTR,  // Сразу скопировать данные с хоста
                                   sizeof(float) * N * M, // Размер буфера в байтах
                                   m_A.data(),            // Указатель на данные A на хосте
                                   &err);                 // Код ошибки
    checkError(err, "clCreateBuffer d_matA");             // Проверка создания буфера A

    cl_mem d_matB = clCreateBuffer(context,               // Контекст
                                   CL_MEM_READ_ONLY |     // Только для чтения
                                   CL_MEM_COPY_HOST_PTR,  // Копировать данные при создании
                                   sizeof(float) * M * K, // Размер буфера B
                                   m_B.data(),            // Указатель на данные B
                                   &err);                 // Код ошибки
    checkError(err, "clCreateBuffer d_matB");             // Проверка создания буфера B

    cl_mem d_matC = clCreateBuffer(context,               // Контекст
                                   CL_MEM_WRITE_ONLY,     // Буфер только для записи в ядре
                                   sizeof(float) * N * K, // Размер буфера C
                                   nullptr,               // Без начальных данных
                                   &err);                 // Код ошибки
    checkError(err, "clCreateBuffer d_matC");             // Проверка создания буфера C

    // Аргументы ядра
    err = clSetKernelArg(kernelMul, 0, sizeof(cl_mem), &d_matA); // Аргумент 0: буфер A
    checkError(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernelMul, 1, sizeof(cl_mem), &d_matB); // Аргумент 1: буфер B
    checkError(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernelMul, 2, sizeof(cl_mem), &d_matC); // Аргумент 2: буфер C
    checkError(err, "clSetKernelArg 2");
    err = clSetKernelArg(kernelMul, 3, sizeof(int), &N);         // Аргумент 3: число строк A / C
    checkError(err, "clSetKernelArg 3");
    err = clSetKernelArg(kernelMul, 4, sizeof(int), &M);         // Аргумент 4: общая размерность (столбцы A / строки B)
    checkError(err, "clSetKernelArg 4");
    err = clSetKernelArg(kernelMul, 5, sizeof(int), &K);         // Аргумент 5: число столбцов B / C
    checkError(err, "clSetKernelArg 5");

    // Глобальная рабочая группа под размер матрицы C
    size_t globalWorkSize[2] = {
        static_cast<size_t>(K), // Размер по оси X = количество столбцов C
        static_cast<size_t>(N)  // Размер по оси Y = количество строк C
    };

    auto gpu_start = chrono::high_resolution_clock::now(); // Старт таймера для GPU

    err = clEnqueueNDRangeKernel(queue,           // Очередь команд
                                 kernelMul,       // Ядро для запуска
                                 2,               // Размерность ND-диапазона (2D)
                                 nullptr,         // Смещение (offset) по умолчанию
                                 globalWorkSize,  // Глобальный размер сетки потоков
                                 nullptr,         // Локальный размер (пусть рантайм подберёт)
                                 0,               // Кол-во событий в wait-list (нет)
                                 nullptr,         // Указатель на события для ожидания (нет)
                                 nullptr);        // Указатель на событие завершения (не нужно)
    checkError(err, "clEnqueueNDRangeKernel");   // Проверяем успешность постановки ядра в очередь

    clFinish(queue);                             // Ждём завершения всех команд в очереди

    auto gpu_end = chrono::high_resolution_clock::now(); // Фиксируем время завершения GPU-вычислений
    double gpu_ms = chrono::duration<double, milli>(gpu_end - gpu_start).count(); // Считаем время GPU в миллисекундах

    // Читаем результат с устройства
    err = clEnqueueReadBuffer(queue,             // Очередь команд
                              d_matC,            // Буфер на устройстве (источник)
                              CL_TRUE,           // Блокирующее чтение (ждём завершения внутри вызова)
                              0,                 // Смещение в буфере
                              sizeof(float) * N * K, // Размер считываемых данных
                              m_C_gpu.data(),    // Целевой буфер на хосте
                              0,                 // Кол-во событий в wait-list
                              nullptr,           // Список событий (нет)
                              nullptr);          // Событие по завершении (не нужно)
    checkError(err, "clEnqueueReadBuffer");      // Проверяем успешность чтения

    cout << "GPU time (OpenCL): " << gpu_ms << " ms" << endl; // Логируем время работы на GPU

    // Проверка корректности
    double maxDiff = 0.0;                        // Переменная для отслеживания максимального расхождения
    for (int i = 0; i < N * K; ++i) {           // Проходим по всем элементам матрицы C
        double diff = fabs(static_cast<double>(m_C_cpu[i]) - // Разница между значениями CPU
                           static_cast<double>(m_C_gpu[i])); // и GPU для элемента i
        if (diff > maxDiff) {                   // Если текущая разница больше максимальной
            maxDiff = diff;                     // Обновляем максимум
        }
    }

    cout << "Max |CPU - GPU| difference: " << maxDiff << endl;         // Логируем максимальное расхождение
    cout << "Sample: CPU C[0] = " << m_C_cpu[0]                       // Выводим пример значения с CPU
         << ", GPU C[0] = " << m_C_gpu[0] << endl;                    // и соответствующее значение с GPU

    // 4. Очистка ресурсов
    clReleaseMemObject(d_matA);                // Освобождаем буфер A на устройстве
    clReleaseMemObject(d_matB);                // Освобождаем буфер B на устройстве
    clReleaseMemObject(d_matC);                // Освобождаем буфер C на устройстве
    clReleaseKernel(kernelMul);                // Освобождаем объект ядра
    clReleaseProgram(program);                 // Освобождаем объект программы
    clReleaseCommandQueue(queue);              // Освобождаем очередь команд
    clReleaseContext(context);                 // Освобождаем контекст OpenCL

    return 0;                                  // Завершение программы
}
