#include <iostream>     
#include <chrono>     
#include <omp.h>        
#include <cstdlib>     

using namespace std;
using namespace std::chrono;

int main() {
    // Настройка параметров

    const int N = 5000000; // Размер массива

    // Динамическое выделение памяти под массив

    int* arr = new (nothrow) int[N]; // new (nothrow) возвращает nullptr при ошибке выделения

    // Проверка успешного выделения памяти
    
    if (!arr) {
        cerr << "Memory allocation error" << endl; // Вывод ошибки
        return 1;                                   // Завершение программы с кодом ошибки
    }

    // Инициализация генератора случайных чисел

    srand(time(0)); // Инициализация srand текущим временем

    // Заполнение массива случайными числами от 1 до 100

    for (int i = 0; i < N; ++i) {
        arr[i] = (rand() % 100) + 1; // rand() % 100 -> [0..99], +1 -> [1..100]
    }

    // Вывод информации о массиве

    cout << "Size of the array: " << N << " elements" << endl;
    cout << "--------------------------------------------" << endl;

    // Последовательное вычисление среднего

    auto start_seq = high_resolution_clock::now(); // Засекаем начало времени

    long long sum_seq = 0; // Переменная для суммы, long long чтобы избежать переполнения
    for (int i = 0; i < N; ++i) {
        sum_seq += arr[i];  // Последовательно суммируем все элементы массива
    }

    double avg_seq = static_cast<double>(sum_seq) / N; // Вычисляем среднее значение

    auto end_seq = high_resolution_clock::now(); // Засекаем конец времени
    double time_seq = duration<double, milli>(end_seq - start_seq).count(); // Время в миллисекундах

    // Вывод результатов последовательного вычисления

    cout << "Sequentially:" << endl;
    cout << "  Average value: " << avg_seq << endl;
    cout << "  Lead time: " << time_seq << " ms" << endl;

    // Параллельное вычисление среднего

    auto start_par = high_resolution_clock::now(); // Начало замера времени

    long long sum_par = 0; // Общая сумма для параллельного вычисления

    // Параллельный цикл с редукцией '+' для суммирования

    #pragma omp parallel for reduction(+:sum_par)
    for (int i = 0; i < N; ++i) {
        sum_par += arr[i]; // Каждый поток добавляет свою часть к локальной сумме
    }

    double avg_par = static_cast<double>(sum_par) / N; // Вычисляем среднее значение

    auto end_par = high_resolution_clock::now(); // Конец замера времени
    double time_par = duration<double, milli>(end_par - start_par).count(); // Время в миллисекундах

    // Вывод результатов параллельного вычисления
    cout << "\nParallel (OpenMP) (OpenMP):" << endl;
    cout << "  Average value: " << avg_par << endl;
    cout << "  Lead time: " << time_par << " ms" << endl;

    // Сравнение результатов

    cout << "--------------------------------------------" << endl;
    cout << "Boost: in " << time_seq / time_par << "x" << endl; // Соотношение времени последовательного и параллельного

    // Освобождение памяти

    delete[] arr; // Обязательно освобождаем динамически выделенную память
    arr = nullptr; // Хорошая практика — обнулять указатель после удаления

    return 0; // Завершение программы
}
