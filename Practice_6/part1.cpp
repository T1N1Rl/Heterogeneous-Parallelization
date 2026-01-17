#include <iostream>        
#include <vector>          
#include <fstream>         
#include <chrono>          
#include <CL/cl.h>         

using namespace std;

// Функция для чтения OpenCL-ядра из внешнего файла
string readKernel(const string& fileName) {
    ifstream file(fileName);                                                        // Открываем файл для чтения
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());  // Считываем все содержимое файла в одну строку и возвращаем её
}

int main() {
    // 1. Константы и подготовка данных
    const int vecSize = 1000000;      // Размер векторов (1000000 элементов)
    vector<float> h_A(vecSize, 1.0f); // Создаем вектор A на CPU и заполняем единицами
    vector<float> h_B(vecSize, 2.0f); // Создаем вектор B на CPU и заполняем двойками
    vector<float> h_C(vecSize, 0.0f); // Создаем вектор C на CPU для записи результата

    // 2. Инициализация OpenCL платформы и устройства
    cl_platform_id platform;              // Идентификатор платформы
    clGetPlatformIDs(1, &platform, NULL); // Получаем доступную платформу

    cl_device_id device; // Идентификатор конкретного устройства
    // Для замера на CPU замените CL_DEVICE_TYPE_GPU на CL_DEVICE_TYPE_CPU
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Создаем среду, в которой будут работать устройства
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    // Создаем очередь команд
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // 3. Компиляция ядра из файла
    string sourceStr = readKernel("kernels.cl"); // Считываем текст кода ядра из файла
    const char* source = sourceStr.c_str();      // Преобразуем строку в понятный для C формат
    // Создаем "программу" OpenCL из исходного кода
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    // Компилируем программу под наше конкретное устройство
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // 4. Создание буферов в видеопамяти (VRAM)
    // d_vecA: для чтения, копируем туда данные из h_A
    cl_mem d_vecA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                   sizeof(float) * vecSize, h_A.data(), NULL);
    // d_vecB: для чтения, копируем туда данные из h_B
    cl_mem d_vecB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                   sizeof(float) * vecSize, h_B.data(), NULL);
    // d_vecC: для записи результата, изначально пустой
    cl_mem d_vecC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                   sizeof(float) * vecSize, NULL, NULL);

    // 5. Настройка аргументов ядра
    // Создаем объект "ядро" на основе скомпилированной функции "vector_add"
    cl_kernel kernelAdd = clCreateKernel(program, "vector_add", NULL);
    // Привязываем буферы к аргументам функции ядра (0, 1, 2 — порядковые номера в kernel.cl)
    clSetKernelArg(kernelAdd, 0, sizeof(cl_mem), &d_vecA);
    clSetKernelArg(kernelAdd, 1, sizeof(cl_mem), &d_vecB);
    clSetKernelArg(kernelAdd, 2, sizeof(cl_mem), &d_vecC);

    // 6. Выполнение и замер времени
    size_t globalVec = vecSize;                        // Указываем общее количество потоков
    auto start = chrono::high_resolution_clock::now(); // Фиксируем время перед запуском

    // Отправляем ядро на выполнение
    clEnqueueNDRangeKernel(queue, kernelAdd, 1, NULL, &globalVec, NULL, 0, NULL, NULL);
    clFinish(queue); // Ожидание завершения

    auto end = chrono::high_resolution_clock::now(); // Фиксируем время после завершения

    // 7. Вывод результата
    cout << "Vector Size: " << vecSize << endl; // Выводим размер вектора
    // Вычисляем разницу во времени и переводим её в миллисекунды
    cout << "Execution Time: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

    // 8. Очистка ресурсов
    clReleaseMemObject(d_vecA); // Удаляем буфер A
    clReleaseMemObject(d_vecB); // Удаляем буфер В
    clReleaseMemObject(d_vecC); // Удаляем буфер С
    clReleaseKernel(kernelAdd); // Удаляем ядро
    clReleaseProgram(program);  // Удаляем программы
    clReleaseCommandQueue(queue); // Удаляем очередь
    clReleaseContext(context);    // Удаляем среду
 
    return 0; // Завершение программы
}