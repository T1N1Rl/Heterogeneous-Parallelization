#include <iostream>              
#include <random>               
#include <chrono>               

using namespace std;            

int main() {                  

    // Объявление констат

    const int N = 50000;          // Размер массива
    const int RAND_MIN_VAL = 1;   // Минимальное значение случайных чисел
    const int RAND_MAX_VAL = 100; // Максимальное значение случайных чисел

    cout << "--- Task 1: Dynamic memory and average value\n"; // Вывод заголовка программы

    // Динамическое выделение памяти

    int* a = new (nothrow) int[N]; // Выделяем динамическую память под массив из N элементов
                                  // nothrow означает, что при ошибке new вернёт nullptr, а не завершит программу

    if (!a) {                     // Проверяем, удалось ли выделить память
        cerr << "Error: failed to allocate memory for the array.\n"; // Сообщение об ошибке
        return 1;                 // Завершаем программу с кодом ошибки
    }

    // Заполнение массива случаянными числами

    random_device rd;             // Источник случайных данных для инициализации генератора
    mt19937 gen(rd());            // Генератор случайных чисел
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL); // Равномерное распределение целых чисел от 1 до 100

    for (int i = 0; i < N; ++i) { // Цикл для заполнения массива
        a[i] = dist(gen);         // Генерируем случайное число и записываем его в массив
    }

    cout << "An array of " << N << " elements has been successfully created and filled in.\n"; // Сообщение об успешном заполнении массива

    // Вычисление среднего значения

    long long sum = 0;            // Переменная для хранения суммы элементов массива
                                  // long long используется, чтобы избежать переполнения

    for (int i = 0; i < N; ++i) { // Цикл для суммирования элементов массива
        sum += a[i];              // Добавляем текущий элемент массива к сумме
    }

    double average = static_cast<double>(sum) / N; // Приводим сумму к double и вычисляем среднее значение

    cout << "Average value of the elements: " << average << endl; // Вывод среднего значения массива

    // Освобождение динамической памяти

    delete[] a;                  // Освобождаем ранее выделенную динамическую память

    cout << "The memory from under the array (delete[]) has been successfully released.\n"; // Сообщение об освобождении памяти

    return 0;                    // Успешное завершение программы
}
