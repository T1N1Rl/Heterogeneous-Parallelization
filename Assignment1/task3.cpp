#include <iostream>   
#include <vector>     
#include <chrono>     
#include <omp.h>      
#include <cstdlib>    
#include <ctime>     

using namespace std;      
using namespace std::chrono; 

int main() {
    // Настройка параметров

    const int N = 1000000;       // Размер массива
    int* arr = new int[N];        // Динамически выделяем память для массива в куче

    srand(time(0));               // Инициализация генератора случайных чисел текущим временем
    for (int i = 0; i < N; ++i) { // Заполнение массива случайными числами
        arr[i] = rand();          // Каждый элемент массива получает случайное целое
    }

    cout << "Size of the array: " << N << endl; // Выводим размер массива для информации

    // Последовательный поиск

    auto start_seq = high_resolution_clock::now(); // Засекаем время начала последовательного поиска

    int min_seq = arr[0];          // Инициализация минимального значения первым элементом
    int max_seq = arr[0];          // Инициализация максимального значения первым элементом

    for (int i = 1; i < N; ++i) { // Проходим по всем элементам массива
        if (arr[i] < min_seq)      // Если текущий элемент меньше текущего min
            min_seq = arr[i];      // Обновляем минимальное значение
        if (arr[i] > max_seq)      // Если текущий элемент больше текущего max
            max_seq = arr[i];      // Обновляем максимальное значение
    }

    auto end_seq = high_resolution_clock::now(); // Засекаем время окончания последовательного поиска
    double time_seq = duration<double, milli>(end_seq - start_seq).count(); // Вычисляем время в миллисекундах

    // Параллельный поиск

    auto start_par = high_resolution_clock::now(); // Засекаем время начала параллельного поиска

    int min_par = arr[0];          // Инициализация минимального значения
    int max_par = arr[0];          // Инициализация максимального значения

    // Параллельный цикл OpenMP с редукцией:
    // reduction(min:min_par) — после завершения всех потоков выбирается минимальное значение среди всех локальных min
    // reduction(max:max_par) — аналогично для максимального значения
    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < N; ++i) { 
        if (arr[i] < min_par)      // Если элемент меньше локального min потока
            min_par = arr[i];      // Обновляем локальный min
        if (arr[i] > max_par)      // Если элемент больше локального max потока
            max_par = arr[i];      // Обновляем локальный max
    }

    auto end_par = high_resolution_clock::now(); // Засекаем время окончания параллельного поиска
    double time_par = duration<double, milli>(end_par - start_par).count(); // Время выполнения в миллисекундах

    //Вывод результатов 

    cout << "\n--- Sequential version ---" << endl; 
    cout << "Min: " << min_seq << ", Max: " << max_seq << endl; // Вывод найденных min и max
    cout << "Time: " << time_seq << " ms" << endl;            // Время последовательного поиска

    cout << "\n--- Parallel version ---" << endl; 
    cout << "Min: " << min_par << ", Max: " << max_par << endl; // Вывод min и max, найденных параллельно
    cout << "Time: " << time_par << " ms" << endl;             // Время параллельного поиска

    cout << "\n--- Comparison ---" << endl;
    if (time_seq > time_par) { // Если последовательная версия медленнее
        cout << "Boost: in " << time_seq / time_par << " time(s)" << endl; // Вывод ускорения
    } else { // Если параллельная версия не ускорила
        cout << "The sequential version turned out to be faster (due to the small N or overhead)" << endl;
    }

    delete[] arr; // Освобождаем динамически выделенную память массива
    return 0;     // Возвращаем 0 
}
