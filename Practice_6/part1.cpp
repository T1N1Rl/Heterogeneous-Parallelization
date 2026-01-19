#include <iostream>        
#include <vector>          
#include <fstream>         
#include <chrono>         
#include <CL/cl.h>         

using namespace std;

// Функция для чтения OpenCL-ядра из внешнего файла
string readKernel(const string& fileName) {
    ifstream file(fileName);                         // Открыть файл с ядром
    return string((istreambuf_iterator<char>(file)), // Прочитать весь файл в строку
                  istreambuf_iterator<char>());
}

// Простая проверка ошибок OpenCL
void checkError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {                         // Если ошибка
        cerr << "OpenCL error (" << err << "): " << msg << endl; //Лог ошибки
        exit(1);                                     // Аварийное завершение
    }
}

int main() {
    const int vecSize = 1'000'000;                   // размер вектора 1M элементов

    vector<float> h_A(vecSize, 1.0f);                // CPU-вектор A, заполнен 1
    vector<float> h_B(vecSize, 2.0f);                // CPU-вектор B, заполнен 2
    vector<float> h_C(vecSize, 0.0f);                // CPU-вектор С, пустой

    // CPU
    auto cpu_start = chrono::high_resolution_clock::now(); // Cтарт таймера

    for (int i = 0; i < vecSize; ++i)                 // Простой цикл на CPU
        h_C[i] = h_A[i] + h_B[i];                     // Сложение

    auto cpu_end = chrono::high_resolution_clock::now();   // Конец таймера
    double cpu_ms = chrono::duration<double, milli>(cpu_end - cpu_start).count(); // Вычислить ms

    cout << "Vector size: " << vecSize << endl;       // Вывести размер данных
    cout << "CPU time: " << cpu_ms << " ms" << endl;  // Вывести время CPU
    cout << "CPU check: C[0] = " << h_C[0] << " (ожидается 3)" << endl; // контроль

    fill(h_C.begin(), h_C.end(), 0.0f);               // Очистка результата перед GPU

    // GPU
    cl_int err;                                       // Код возврата OpenCL

    cl_platform_id platform = nullptr;                // Дескриптор платформы
    err = clGetPlatformIDs(1, &platform, nullptr);    // Выбрать первую доступную платформу
    checkError(err, "clGetPlatformIDs");

    cl_device_id device = nullptr;                    // Дескриптор устройства (GPU)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr); // Выбрать GPU
    checkError(err, "clGetDeviceIDs");

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); // Создать контекст
    checkError(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err); // Очередь команд
    checkError(err, "clCreateCommandQueue");

    string sourceStr = readKernel("kernels.cl");       // Загрузить текст ядра
    const char* source = sourceStr.c_str();            // Получить C-строку
    size_t sourceSize = sourceStr.size();              // Размер текста

    cl_program program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &err); // Создать программу
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr); // Компиляция под устройство
    if (err != CL_SUCCESS) {                            // Если ошибка компиляции
        size_t log_size;                                // Размер лога
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size); // Получить размер
        vector<char> log(log_size);                     // Буфер для лога
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr); // Получить лог
        cerr << "Build log:\n" << log.data() << endl;    //Вывести лог
        checkError(err, "clBuildProgram");
    }

    cl_kernel kernelAdd = clCreateKernel(program, "vector_add", &err); // Создать kernel из функции
    checkError(err, "clCreateKernel");

    cl_mem d_vecA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * vecSize, h_A.data(), &err); // Буфер A на GPU
    checkError(err, "clCreateBuffer d_vecA");

    cl_mem d_vecB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * vecSize, h_B.data(), &err); // Буфер B на GPU
    checkError(err, "clCreateBuffer d_vecB");

    cl_mem d_vecC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * vecSize, nullptr, &err); // Буфер C на GPU
    checkError(err, "clCreateBuffer d_vecC");

    err = clSetKernelArg(kernelAdd, 0, sizeof(cl_mem), &d_vecA); // Аргумент 0: A
    checkError(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernelAdd, 1, sizeof(cl_mem), &d_vecB); // Аргумент 1: B
    checkError(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernelAdd, 2, sizeof(cl_mem), &d_vecC); // Аргумент 2: C
    checkError(err, "clSetKernelArg 2");

    size_t globalWorkSize = static_cast<size_t>(vecSize); // Число потоков = размер вектора

    auto gpu_start = chrono::high_resolution_clock::now(); // Старт таймера GPU

    err = clEnqueueNDRangeKernel(queue, kernelAdd, 1, nullptr,
                                 &globalWorkSize, nullptr, 0, nullptr, nullptr); // Отправить kernel на GPU
    checkError(err, "clEnqueueNDRangeKernel");

    clFinish(queue);                                  // Дождаться окончания вычислений

    auto gpu_end = chrono::high_resolution_clock::now(); // Стоп таймера GPU
    double gpu_ms = chrono::duration<double, milli>(gpu_end - gpu_start).count(); // Время GPU

    err = clEnqueueReadBuffer(queue, d_vecC, CL_TRUE, 0,
                              sizeof(float) * vecSize, h_C.data(),
                              0, nullptr, nullptr);    // Копирование результата с GPU → CPU
    checkError(err, "clEnqueueReadBuffer");

    cout << "GPU time (OpenCL): " << gpu_ms << " ms" << endl; // Вывести время GPU
    cout << "GPU check: C[0] = " << h_C[0] << " (ожидается 3)" << endl; // Контроль корректности

    clReleaseMemObject(d_vecA);                        // Освобождение буфера A
    clReleaseMemObject(d_vecB);                        // Освобождение буфера B
    clReleaseMemObject(d_vecC);                        // Освобождение буфера C
    clReleaseKernel(kernelAdd);                        // Освобождение ядра
    clReleaseProgram(program);                         // Освобождение программы
    clReleaseCommandQueue(queue);                      // Освобождение очереди
    clReleaseContext(context);                         // Освобождение контекста

    return 0;                                          // Завершение программы
}
