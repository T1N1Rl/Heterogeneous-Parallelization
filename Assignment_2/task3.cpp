#include <iostream>       
#include <vector>         
#include <chrono>           
#include <omp.h>           
#include <cstdlib>        
#include <ctime>         

using namespace std;       
using namespace chrono;    

// Функция заполнения массива случайными числами
void fillArray(vector<int>& arr) {          // Принимаем массив по ссылке
    for (int i = 0; i < arr.size(); i++) {  // Проходим по всем элементам массива
        arr[i] = rand() % 10000;             // Записываем случайное число от 0 до 9999
    }
}

// Последовательная сортировка
void selectionSortSequential(vector<int>& arr) { // Функция последовательной сортировки
    int n = arr.size();                           // Получаем размер массива

    for (int i = 0; i < n - 1; i++) {             // Внешний цикл по позициям массива
        int minIndex = i;                         // Считаем текущий элемент минимальным

        for (int j = i + 1; j < n; j++) {         // Ищем минимум в оставшейся части массива
            if (arr[j] < arr[minIndex]) {         // Если найден элемент меньше текущего минимума
                minIndex = j;                     // Обновляем индекс минимума
            }
        }

        swap(arr[i], arr[minIndex]);              // Меняем местами текущий элемент и минимум
    }
}

// Параллельная сортировка
void selectionSortParallel(vector<int>& arr) {    // Функция параллельной сортировки
    int n = arr.size();                            // Получаем размер массива

    for (int i = 0; i < n - 1; i++) {              // Внешний цикл (НЕ параллелится)
        int minIndex = i;                          // Глобальный индекс минимума

        #pragma omp parallel for                  // Распараллеливаем поиск минимума
        for (int j = i + 1; j < n; j++) {          // Каждый поток обрабатывает свою часть массива
            if (arr[j] < arr[minIndex]) {          // Если найден меньший элемент
                #pragma omp critical               // Заходим в критическую секцию
                {
                    if (arr[j] < arr[minIndex]) {  // Повторная проверка (защита от гонок)
                        minIndex = j;              // Обновляем глобальный минимум
                    }
                }
            }
        }

        swap(arr[i], arr[minIndex]);               // Меняем текущий элемент с найденным минимумом
    }
}


void runTest(int size) {                           // Функция тестирования для заданного размера
    vector<int> data(size);                        // Создаём массив нужного размера
    fillArray(data);                               // Заполняем массив случайными числами

    vector<int> dataCopy = data;                   // Создаём копию массива для честного сравнения

    cout << "Array sizw: " << size << endl;    // Выводим размер массива

    // Последовательная версия 
    auto startSeq = high_resolution_clock::now();  // Засекаем начало времени
    selectionSortSequential(data);                 // Запускаем последовательную сортировку
    auto endSeq = high_resolution_clock::now();    // Засекаем конец времени

    double seqTime = duration<double, milli>(endSeq - startSeq).count(); // Вычисляем время
    cout << "Sequential sorting: " << seqTime << " ms" << endl;  // Выводим время

    // Параллельная версия
    auto startPar = high_resolution_clock::now();  // Засекаем начало времени
    selectionSortParallel(dataCopy);               // Запускаем параллельную сортировку
    auto endPar = high_resolution_clock::now();    // Засекаем конец времени

    double parTime = duration<double, milli>(endPar - startPar).count();  // Вычисляем время
    cout << "Parallel sorting:     " << parTime << " ms" << endl;  // Выводим время

    cout << "Boost: " << seqTime / parTime << "x" << endl;            // Выводим ускорение
    cout << "----------------------------------------" << endl;              // Разделитель
}


int main() {                                       // Точка входа в программу
    srand(time(0));                                // Инициализируем генератор случайных чисел

    #ifdef _OPENMP                                // Проверяем, поддерживается ли OpenMP
        cout << "OpenMP is active. Threads: "
             << omp_get_max_threads() << endl;     // Выводим количество потоков
    #endif

    runTest(1000);                                 // Тест для массива из 1000 элементов
    runTest(10000);                                // Тест для массива из 10000 элементов

    return 0;                                      // Завершаем программу 
}
