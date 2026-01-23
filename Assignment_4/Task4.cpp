#include <mpi.h>     
#include <iostream>  
#include <vector>    
#include <random>     


// Задание 4 (25 баллов)
// Реализуйте распределённую программу с использованием MPI для обработки массива
// данных. Разделите массив между процессами, выполните вычисления локально и
// соберите результаты. Проведите замеры времени выполнения для 2, 4 и 8 процессов.

int main(int argc, char** argv) {
    // Инициализация среды MPI (создает процессы)
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    // Получаем уникальный номер текущего процесса (от 0 до size-1)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получаем общее количество запущенных процессов
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Общий размер массива, который нужно обработать (по умолчанию 100 млн элементов)
    long long N = (argc > 1) ? std::atoll(argv[1]) : 100000000LL;

    // Только главный процесс (rank 0) выводит начальную информацию
    if (rank == 0) {
        std::cout << "Task 4: MPI distributed sum, total N = " << N
                  << ", processes = " << size << std::endl;
    }

    // РАСПРЕДЕЛЕНИЕ РАБОТЫ (Data Decomposition)
    // Делим общее число N на количество процессов
    long long base = N / size; // Целая часть
    long long rem  = N % size; // Остаток (если N не делится нацело)
    
    // Если есть остаток, первые 'rem' процессов берут на 1 элемент больше
    long long local_N = base + (rank < rem ? 1 : 0);

    // Каждый процесс создает свой локальный кусок данных
    std::vector<double> local_data(local_N);
    // Инициализируем генератор случайных чисел (зерно зависит от rank, чтобы данные были разными)
    std::mt19937_64 gen(42 + rank);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Заполняем локальный массив данными
    for (long long i = 0; i < local_N; ++i) {
        local_data[i] = dist(gen);
    }

    // Синхронизация: ждем, пока все процессы закончат генерацию данных
    MPI_Barrier(MPI_COMM_WORLD);
    // Засекаем время начала вычислений
    double t_start = MPI_Wtime();

    // Локальные вычисления
    double local_sum = 0.0;
    for (long long i = 0; i < local_N; ++i) {
        local_sum += local_data[i];
    }

    //Сбор результатов
    double global_sum = 0.0;
    // MPI_Reduce собирает local_sum со всех процессов, суммирует их (MPI_SUM) 
    // и записывает результат в global_sum только на процессе 0.
    
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Фиксируем время окончания вычислений
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    // Главный процесс выводит итоговый результат
    if (rank == 0) {
        double mean = global_sum / static_cast<double>(N); // Считаем среднее значение
        std::cout << "Result mean = " << mean
                  << ", time = " << elapsed << " s\n";
    }

    // Завершение работы MPI
    MPI_Finalize();
    return 0;
}