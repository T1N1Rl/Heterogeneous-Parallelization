#include <mpi.h>     
#include <iostream>  
#include <vector>     
#include <random>     


// Задание 4. Анализ масштабируемости распределённой программы (MPI)
int main(int argc, char** argv) {
    // Инициализация среды MPI
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    // Получаем номер текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получаем общее количество процессов (P)
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Параметр N: 
    // В режиме strong scaling — это общий размер массива на все процессы.
    // В режиме weak scaling — это размер данных на один процесс.
    long long N = (argc > 1) ? std::atoll(argv[1]) : 100000000LL; // По умолчанию 10^8

    // Выбор режима масштабируемости (0 - Strong, 1 - Weak)
    int mode = (argc > 2) ? std::atoi(argv[2]) : 0;

    long long local_N = 0;
    if (mode == 0) {
        // Strong Scaling
        // Общий объем работы фиксирован. С ростом числа процессов нагрузка на каждый падает.
        long long base = N / size;
        long long rem  = N % size;
        local_N = base + (rank < rem ? 1 : 0); // Распределяем остаток строк
    } else {
        // Weak Scaling
        // Общий объем работы растет пропорционально числу процессов. 
        // Нагрузка на каждый процесс (local_N) остается постоянной.
        local_N = N;
    }

    // Вывод информации о запуске (только процессом 0)
    if (rank == 0) {
        if (mode == 0)
            std::cout << "MPI aggregation, strong scaling, total N = " << N << "\n";
        else
            std::cout << "MPI aggregation, weak scaling, local N = " << local_N << "\n";
    }

    // Создаем локальный массив для каждого процесса
    std::vector<double> local_data(local_N);
    // Инициализируем генератор (зерно зависит от rank, чтобы данные были разными)
    std::mt19937_64 gen(42 + rank);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Заполняем локальную часть массива
    for (long long i = 0; i < local_N; ++i) {
        local_data[i] = dist(gen);
    }

    // Синхронизация всех процессов перед замером времени
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Локальный расчет: каждый процесс считает сумму своей части
    double local_sum = 0.0;
    for (long long i = 0; i < local_N; ++i) {
        local_sum += local_data[i];
    }

    // Глобальная агрегация: собираем суммы со всех процессов
    double global_sum = 0.0;
    // MPI_Allreduce собирает данные со всех и рассылает результат всем обратно.
    // 
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Фиксируем время окончания
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    // Вывод итогов
    if (rank == 0) {
        // Считаем общее количество обработанных элементов
        long long global_N = (mode == 0) ? N : (local_N * size);
        double mean = global_sum / (double)global_N;
        std::cout << "P = " << size << ", time = " << elapsed
                  << " s, mean = " << mean << std::endl;
    }

    // Завершение работы с MPI
    MPI_Finalize();
    return 0;
}