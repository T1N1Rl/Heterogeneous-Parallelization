#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <CL/cl.h>

using namespace std;

// Функция для чтения OpenCL-ядра
string readKernel(const string& fileName) {
    ifstream file(fileName);                                                        // Открываем файл для чтения
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());  // Считываем все содержимое файла в одну строку и возвращаем её
}

int main() {
    // 1. Параметры матриц (A[NxM] * B[MxK] = C[NxK])
    const int N = 512, M = 512, K = 512; // Задаем размеры: 512x512

    // Подготовка данных на CPU
    vector<float> m_A(N * M, 1.5f); // Матрица A заполнена числами 1.5
    vector<float> m_B(M * K, 0.5f); // Матрица B заполнена числами 0.5
    vector<float> m_C(N * K, 0.0f); // Матрица C (результат) заполнена нулями

    // 2. Инициализация OpenCL
    cl_platform_id platform;               // Идентификатор платформы (NVIDIA, Intel, AMD)
    clGetPlatformIDs(1, &platform, NULL);  // Получаем список доступных платформ
    cl_device_id device;                   // Идентификатор устройства (видеокарты)
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); // Ищем именно GPU
    
    // Создаем контекст — область, в которой OpenCL управляет памятью и ядрами
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    // Создаем командную очередь для отправки задач на GPU
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // 3. Сборка программы
    string sourceStr = readKernel("kernels.cl"); // Считываем исходный код ядра
    const char* source = sourceStr.c_str();      // Приводим к формату C-строки
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL); // Создаем объект программы из исходного текста
    // Компилируем программу специально под видеокарту
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // 4. Создание буферов в памяти GPU
    // CL_MEM_COPY_HOST_PTR сразу копирует данные в видеопамять при создании
    cl_mem d_matA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * M, m_A.data(), NULL);
    cl_mem d_matB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * K, m_B.data(), NULL);
    // Для матрицы C создаем буфер только для записи
    cl_mem d_matC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * K, NULL, NULL);

    // 5. Настройка ядра
    // Создаем объект ядра "matrix_mul"
    cl_kernel kernelMul = clCreateKernel(program, "matrix_mul", NULL);
    // Привязываем аргументы ядра (указатели на матрицы и их размеры)
    clSetKernelArg(kernelMul, 0, sizeof(cl_mem), &d_matA); // Матрица A
    clSetKernelArg(kernelMul, 1, sizeof(cl_mem), &d_matB); // Матрица В
    clSetKernelArg(kernelMul, 2, sizeof(cl_mem), &d_matC); // Матрица С
    clSetKernelArg(kernelMul, 3, sizeof(int), &N);         // Число строк A
    clSetKernelArg(kernelMul, 4, sizeof(int), &M);         // Общая сторона (столбцы A / строки B)
    clSetKernelArg(kernelMul, 5, sizeof(int), &K);         // Число столбцов B

    // 6. Запуск вычислений (2D сетка: Строки x Столбцы)
    size_t globalWorkSize[2] = { (size_t)K, (size_t)N }; // K потоков по горизонтали, N по вертикали
    
    auto start = chrono::high_resolution_clock::now(); // Начинаем отсчет времени
    // Запускаем двумерное ядро
    clEnqueueNDRangeKernel(queue, kernelMul, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(queue); // Запускаем двумерное ядро
    auto end = chrono::high_resolution_clock::now();  // Заканчиваем отсчет времени

    // Выводим результаты
    cout << "Matrix Multiplication (" << N << "x" << K << ") completed." << endl;
    cout << "Time: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

    // 7. Очистка ресурсов
    clReleaseMemObject(d_matA); clReleaseMemObject(d_matB); clReleaseMemObject(d_matC);
    clReleaseKernel(kernelMul); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);

    return 0; //Завершение программы
}