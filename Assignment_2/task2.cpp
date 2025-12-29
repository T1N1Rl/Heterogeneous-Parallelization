#include <iostream>             
#include <vector>               
#include <algorithm>            
#include <chrono>               
#include <omp.h>                
#include <ctime>                

using namespace std;            
using namespace std::chrono;    

int main() {                    

    const int N = 10000;         // Размер массива — 10 000 
    vector<int> arr(N);          // Вектор целых чисел размером N

    srand(time(0));              // Инициализация генератора случайных чисел текущим временем

    for (int i = 0; i < N; ++i) {// Цикл заполнения массива
        arr[i] = rand() % 100000;// Запись случайных чисел от 0 до 99 999
    }

    cout << "Array size: " << N << endl; // Выводим размер массива
    cout << "------------------------------------------" << endl; // Разделительная линия

    // Последовательная реализация

    int min_seq = arr[0];        // Инициализация минимальное значение первым элементом
    int max_seq = arr[0];        // Инициализация максимальное значение первым элементом

    auto start_seq = high_resolution_clock::now(); // Запоминаем время начала последовательного алгоритма

    for (int i = 0; i < N; ++i) {// Последовательный проход по массиву
        if (arr[i] < min_seq)    // Если текущий элемент меньше текущего минимума
            min_seq = arr[i];    // Обновляем минимум
        if (arr[i] > max_seq)    // Если текущий элемент больше текущего максимума
            max_seq = arr[i];    // Обновляем максимум
    }

    auto end_seq = high_resolution_clock::now(); // Фиксируем время окончания последовательного алгоритма
    duration<double, milli> time_seq = end_seq - start_seq; // Вычисляем время выполнения в миллисекундах

    cout << "[Sequential] Min: " << min_seq << ", Max: " << max_seq << endl; // Вывод min и max
    cout << "Time: " << time_seq.count() << " ms" << endl; // Вывод времени выполнения
    cout << "------------------------------------------" << endl; // Разделительная линия

    // Параллельная реализация

    int min_par = arr[0];        // Начальное значение минимума для параллельной версии
    int max_par = arr[0];        // Начальное значение максимума для параллельной версии

    auto start_par = high_resolution_clock::now(); // Запоминаем время начала параллельного алгоритма

    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    // Директива OpenMP:
    // parallel for — распараллеливает цикл
    // reduction(min:min_par) — безопасно вычисляет минимум
    // reduction(max:max_par) — безопасно вычисляет максимум
    for (int i = 0; i < N; ++i) {// Параллельный проход по массиву
        if (arr[i] < min_par)    // Сравниваем элемент с локальным минимумом потока
            min_par = arr[i];    // Обновляем минимум
        if (arr[i] > max_par)    // Сравниваем элемент с локальным максимумом потока
            max_par = arr[i];    // Обновляем максимум
    }

    auto end_par = high_resolution_clock::now(); // Фиксируем время окончания параллельного алгоритма
    duration<double, milli> time_par = end_par - start_par; // Вычисляем время выполнения в миллисекундах

    cout << "[Parallel (OpenMP)] Min: " << min_par << ", Max: " << max_par << endl; // Вывод min и max
    cout << "Time: " << time_par.count() << " ms" << endl; // Вывод времени выполнения
    cout << "------------------------------------------" << endl; // Разделительная линия

    // Сравнение результатов

    cout << "Comparison results:" << endl; // Заголовок блока сравнения

    if (time_seq.count() > time_par.count()) { // Если параллельная версия быстрее
        cout << "The parallel version is faster in"
             << time_seq.count() / time_par.count()
             << " time(s)." << endl; // Вывод коэффициента ускорения
    } else {                                   // Если последовательная версия быстрее или равна
        cout << "Sequential version turned out to be faster or equal to the parallel one." << endl;
        cout << "Reason: the small size of the array and the overhead of threads." << endl;
    }

    return 0;                    // Завершаем программу 
}
