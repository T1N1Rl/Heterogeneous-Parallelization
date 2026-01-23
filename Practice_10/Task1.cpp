#include <iostream>
#include <vector>  
#include <random>   
#include <cmath>    
#include <omp.h>    


// Задание 1. Анализ производительности CPU-параллельной программы (OpenMP)
int main(int argc, char** argv) {
    // Определяем размер массива: берем из аргументов командной строки или ставим 10^7 по умолчанию
    long long N = (argc > 1) ? std::atoll(argv[1]) : 10000000LL;

    std::cout << "OpenMP performance test, N = " << N << std::endl;

    // Создаем вектор из N чисел типа double
    std::vector<double> data(N);
    // Инициализируем 64-битный генератор случайных чисел Вихрь Мерсенна зерном 42
    std::mt19937_64 gen(42);
    // Устанавливаем диапазон случайных чисел от 0.0 до 1.0
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Заполняем массив данными в один поток (последовательно)
    for (long long i = 0; i < N; ++i) {
        data[i] = dist(gen);
    }

    // Последовательная версия
    double seq_sum = 0.0;    // Переменная для суммы
    double seq_sum_sq = 0.0; // Переменная для суммы квадратов (нужна для дисперсии)

    // Запоминаем текущее время перед началом расчетов на одном ядре
    double t_seq_start = omp_get_wtime();
    for (long long i = 0; i < N; ++i) {
        double x = data[i];
        seq_sum    += x;     // Накапливаем общую сумму
        seq_sum_sq += x * x; // Накапливаем сумму квадратов элементов
    }
    // Запоминаем время окончания
    double t_seq_end = omp_get_wtime();
    // Вычисляем общую длительность последовательного выполнения в секундах
    double t_seq = t_seq_end - t_seq_start;

    // Вычисляем среднее значение и дисперсию для проверки точности
    double seq_mean = seq_sum / (double)N;
    double seq_var  = seq_sum_sq / (double)N - seq_mean * seq_mean;

    std::cout << "Sequential:\n";
    std::cout << "  mean = " << seq_mean << ", variance = " << seq_var << "\n";
    std::cout << "  time = " << t_seq << " s\n\n";

    // Узнаем, сколько всего логических ядер (потоков) доступно системе
    int max_threads = omp_get_max_threads();
    std::cout << "Max threads available: " << max_threads << "\n";

    // Параллельная версия
    // Запускаем цикл тестов для разного количества потоков: 1, 2, 4, 8...
    for (int threads : {1, 2, 4, 8, 16, 32}) {
        // Если запрашиваемое число потоков больше физически доступных — пропускаем итерацию
        if (threads > max_threads) continue;

        // Явно устанавливаем количество потоков для следующей параллельной секции
        omp_set_num_threads(threads);

        double par_sum = 0.0;    // Локальные переменные для параллельного расчета
        double par_sum_sq = 0.0;

        // Фиксируем время начала параллельной обработки
        double t_par_start = omp_get_wtime();
        
        // Директива параллельного цикла с редукцией. 
        // reduction(+:...) создает копии переменных для каждого потока, а в конце суммирует их.
        #pragma omp parallel for reduction(+:par_sum, par_sum_sq)
        for (long long i = 0; i < N; ++i) {
            double x = data[i];
            par_sum    += x;     // Каждый поток считает свою часть суммы
            par_sum_sq += x * x; // Каждый поток считает свою часть суммы квадратов
        }
        // Фиксируем время окончания параллельной обработки
        double t_par_end = omp_get_wtime();
        double t_par = t_par_end - t_par_start;

        // Повторный расчет статистик для контроля корректности данных
        double mean = par_sum / (double)N;
        double var  = par_sum_sq / (double)N - mean * mean;

        // Вычисляем ускорение (Speedup): во сколько раз параллельная версия быстрее последовательной
        double speedup = t_seq / t_par;

        // Оценка доли параллельной части P по закону Амдала:
        // Закон Амдала: Speedup = 1 / ((1 - P) + P / n)
        // Отсюда вычисляем P — теоретический процент кода, который реально ускорился.
        double P_est = 0.0;
        if (threads > 1 && speedup > 1.0) {
            P_est = (1.0 - 1.0 / speedup) / (1.0 - 1.0 / threads);
        }

        std::cout << "Threads = " << threads << "\n";
        std::cout << "  mean = " << mean << ", variance = " << var << "\n";
        std::cout << "  time = " << t_par << " s, speedup = " << speedup << "\n";
        // Вывод оценки параллельной доли. Если P близко к 1.0 — код идеально параллелен.
        std::cout << "  estimated parallel fraction P ≈ " << P_est << "\n\n";
    }

    std::cout << "Done.\n";
    return 0; // Завершение программы
}