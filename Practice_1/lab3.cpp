#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <limits>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Создает и заполняет динамический массив случайными числами (1-100)
int* createAndFillArray(int N) { // N — размер массива
    int* arr = new (nothrow) int[N]; // // Выделяем память под массив на куче (Heap) с использованием nothrow для безопасной обработки ошибок
    if (!arr) { // Проверка успешности выделения памяти
        cerr << "Error: Failed to allocate memory (N = " << N << ")" << endl;
        return nullptr;
    }

    srand(time(0)); // Иницилизация генератора случайных чисел (1-100)
    for (int i = 0; i < N; ++i) arr[i] = (rand() % 100) + 1; // Числа от 1 до 100
    return arr;
}

// Последовательный расчет среднего значения элементов массива
double calculateAvgSeq(int* arr, int N) {
    if (N <= 0 || !arr) return 0.0; // Проверка на пустой или некорректный массив

    long long sum = 0; // Предотвращание переволнения при суммирование
    for (int i = 0; i < N; ++i) sum += arr[i]; // Последовательное суммирование 
    return (double)sum / N; // Рассчет среднего и возврат результата
}

// Параллельный расчет среднего занчения с использованием OpenMP Reduction
double calculateAvgPar(int* arr, int N, int threads) { 
    if (N <= 0 || !arr) return 0.0; // Проверка на пустой или некорректный массив

    long long sum = 0; // Пременнная, которая пойдет в reduction

#ifdef _OPENMP
    omp_set_num_threads(threads); // Установка ккачества потоков 
    #pragma omp parallel for reduction(+:sum) // Распределяет итерации цикла между потоками для их одновременного выполнения
    for (int i = 0; i < N; ++i) sum += arr[i]; // Каждый поток счиатет свою часть суммы
#else
    for (int i = 0; i < N; ++i) sum += arr[i]; // Последовательное суммирование, если OpenMP не доступна
#endif

    return (double)sum / N; // Расчет среднего значения
}

// Основа 
int main() {

    // Настройка и ввод данных
    int array_size; // Размер массива N
    const int MIN_SIZE = 1000000; // Рекомендуемый минимальнный размер для ввода
    int num_threads = 1; // Переменная хранящая число потоков  

    cout << "\n--- Dynamic Memory and Parallel Summation ---\n";
    cout << "Enter array size N (recommended > 1,000,000): "; 
    if (!(cin >> array_size) || array_size <= 0) { // Ввод размер N с базовой проверкой 
        cerr << "Error: Invalid array size entered.\n";
        return 1;
    }
    if (array_size < MIN_SIZE) // Предупреждение о малом размере
        cout << "WARNING: Size N=" << array_size << " is small; overhead may reduce speedup.\n"; 

#ifdef _OPENMP
    num_threads = omp_get_max_threads();
    cout << "INFO: OpenMP active, available threads: " << num_threads << endl; //Получение максимального доступного числа потоков
#else
    cout << "INFO: OpenMP not active, running single-threaded.\n";
#endif

    cout << "Array Size (N): " << array_size << endl;

    // Создание массивов
    auto t_create_start = chrono::high_resolution_clock::now(); // Засекаем время 
    int* dataArray = createAndFillArray(array_size); // Вызываем функцию создание 
    auto t_create_end = chrono::high_resolution_clock::now(); // Засекаем время конца
    chrono::duration<double, milli> dur_create = t_create_end - t_create_start; // Расчет времени выолнено в миллисекундах

    if (!dataArray) return 1; // Выход при ошибке выделение памяти 
    cout << "1. Array created and filled in " << dur_create.count() << " ms.\n";

    // Последовательный расчет среднего
    auto t_seq_start = chrono::high_resolution_clock::now(); // Начало расчета
    double avg_seq = calculateAvgSeq(dataArray, array_size); // Вызов последовательной функции
    auto t_seq_end = chrono::high_resolution_clock::now(); // Конец расчета
    chrono::duration<double, milli> dur_seq = t_seq_end - t_seq_start; // Время выполнение

    cout << "\n--- Sequential Calculation ---\n";
    cout << "Average Value: " << avg_seq << "\n";
    cout << "Execution Time: " << dur_seq.count() << " ms\n";

    // Параллельный расчет срееднего 
    auto t_par_start = chrono::high_resolution_clock::now(); // Начало расчета
    double avg_par = calculateAvgPar(dataArray, array_size, num_threads); // Вызов параллельной функции
    auto t_par_end = chrono::high_resolution_clock::now(); // Конец расчета
    chrono::duration<double, milli> dur_par = t_par_end - t_par_start; // Время выполнение

    cout << "\n--- Parallel Calculation ---\n";
    cout << "Average Value: " << avg_par << "\n";
    cout << "Execution Time: " << dur_par.count() << " ms\n";

    // Анализ и очистка 
    cout << "\n--- Analysis ---\n";
    double speedup = dur_seq.count() / dur_par.count(); // Расчет ускорения

    cout << "Verification (avg_seq == avg_par): " // Проверка совпадении в последовательного и параллельного расчетов
         << (avg_seq == avg_par ? "Success" : "Failure") << "\n";
    cout << "Speedup: " << speedup << "x\n";

    if (speedup > 1.0) // Выводы
        cout << "CONCLUSION: Parallel summation achieved speedup.\n";
    else
        cout << "CONCLUSION: Thread overhead reduced benefit.\n";

    delete[] dataArray; // Очистка памяти
    cout << "Memory successfully freed (delete[]).\n";

    return 0;
}
