#include <mpi.h>      
#include <iostream>  
#include <vector>     
#include <random>     
#include <cmath>     
#include <algorithm> 


// Задание 2: Распределённое решение системы линейных уравнений методом Гаусса
int main(int argc, char** argv) {
    // Инициализируем среду MPI (создаем процессы, настраиваем сетевое взаимодействие)
    MPI_Init(&argc, &argv); 

    int rank = 0, size = 1;
    // Определяем порядковый номер (ранг) текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    // Определяем общее количество запущенных процессов
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int N = 8; // Размерность матрицы (N x N) по умолчанию
    // Если передан аргумент в командной строке, используем его как размер N
    if (argc >= 2) {
        N = std::atoi(argv[1]); 
        if (N <= 0) N = 8;
    }

    // Проверяем кратность: MPI_Scatter проще всего работает, когда N делится на количество процессов
    if (N % size != 0 && rank == 0) {
        std::cerr << "Warning: N % P != 0, лучше взять N кратным числу процессов.\n";
    }

    // Рассчитываем количество строк на один процесс (округляем вверх)
    int rows_per_proc = (N + size - 1) / size; 
    // Считаем реальное кол-во строк для текущего процесса (последний может получить меньше)
    int local_rows = std::min(rows_per_proc, N - rank * rows_per_proc);
    if (local_rows < 0) local_rows = 0;

    std::vector<double> A; // Контейнер для полной матрицы (заполнится только на rank 0)
    std::vector<double> b; // Контейнер для вектора свободных членов (только на rank 0)

    if (rank == 0) {
        A.resize(N * N); // Выделяем память под N*N элементов
        b.resize(N);     // Выделяем память под N элементов

        std::mt19937 gen(42); // Генератор случайных чисел с фиксированным зерном
        std::uniform_real_distribution<double> dist(1.0, 10.0); // Диапазон чисел от 1 до 10

        // Заполняем матрицу так, чтобы она была диагонально доминирующей (для стабильности Гаусса)
        for (int i = 0; i < N; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                double val = dist(gen);
                A[i * N + j] = val; // Заполняем недиагональные элементы
                row_sum += std::fabs(val);
            }
            A[i * N + i] = row_sum + 1.0; // Диагональный элемент больше суммы остальных в строке
            b[i] = dist(gen); // Заполняем правую часть уравнения
        }
    }

    // Локальные буферы, куда каждый процесс получит свои строки из общей матрицы
    std::vector<double> localA(rows_per_proc * N, 0.0);
    std::vector<double> localb(rows_per_proc, 0.0);

    // 
    // Разрезаем матрицу A на горизонтальные блоки и раздаем процессам
    MPI_Scatter(
        rank == 0 ? A.data() : nullptr, // Откуда берем данные (только на rank 0)
        rows_per_proc * N,              // Сколько элементов отправляем каждому
        MPI_DOUBLE,                     // Тип данных
        localA.data(),                  // Куда принимаем данные на каждом процессе
        rows_per_proc * N,              // Сколько элементов принимаем
        MPI_DOUBLE,                     // Тип принимаемых данных
        0,                              // Кто раздает (root)
        MPI_COMM_WORLD                  // Группа процессов
    );

    // Раздаем вектор b аналогичным образом
    MPI_Scatter(
        rank == 0 ? b.data() : nullptr,
        rows_per_proc,
        MPI_DOUBLE,
        localb.data(),
        rows_per_proc,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Ждем, пока все закончат прием данных
    MPI_Barrier(MPI_COMM_WORLD);
    // Начинаем отсчет времени для прямой фазы Гаусса
    double t_start = MPI_Wtime();

    // ПРЯМОЙ ХОД (Приведение к верхнетреугольному виду)
    for (int k = 0; k < N; ++k) {
        // Определяем, какой процесс владеет "ведущей" строкой k
        int owner = k / rows_per_proc;
        // Порядковый номер строки k внутри локального буфера этого процесса
        int local_k = k - owner * rows_per_proc;

        std::vector<double> pivotRow(N); // Буфер для хранения ведущей строки
        double pivotB = 0.0;             // Элемент вектора b для этой строки

        // Процесс-владелец подготавливает данные для рассылки
        if (rank == owner && local_k >= 0 && local_k < rows_per_proc) {
            double* row_ptr = &localA[local_k * N];
            for (int j = 0; j < N; ++j) {
                pivotRow[j] = row_ptr[j];
            }
            pivotB = localb[local_k];
        }

        // 
        // Рассылаем ведущую строку и элемент b всем остальным процессам
        MPI_Bcast(pivotRow.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivotB, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        double pivot = pivotRow[k]; // Ведущий (диагональный) элемент
        if (std::fabs(pivot) < 1e-12) continue; // Проверка на нулевой элемент

        // Каждый процесс обновляет только те строки, которые лежат ниже текущей ведущей k
        for (int i_local = 0; i_local < rows_per_proc; ++i_local) {
            int i_global = rank * rows_per_proc + i_local;
            // Исключаем обработку строк выше ведущей и пустых строк (если N не кратно P)
            if (i_global <= k || i_global >= N) continue;

            double a_ik = localA[i_local * N + k]; // Элемент, который хотим обнулить
            if (std::fabs(a_ik) < 1e-12) continue;

            double factor = a_ik / pivot; // Коэффициент вычитания
            // Вычитаем ведущую строку из текущей локальной строки
            for (int j = k; j < N; ++j) {
                localA[i_local * N + j] -= factor * pivotRow[j];
            }
            localb[i_local] -= factor * pivotB; // Обновляем правую часть b
        }
    }

    // Собираем результаты: обнуляем глобальные контейнеры на rank 0 перед приемом
    if (rank == 0) {
        A.assign(N * N, 0.0);
        b.assign(N, 0.0);
    }

    // Собираем обработанные строки обратно на главный процесс
    MPI_Gather(
        localA.data(), rows_per_proc * N, MPI_DOUBLE, 
        rank == 0 ? A.data() : nullptr, rows_per_proc * N, MPI_DOUBLE, 
        0, MPI_COMM_WORLD
    );

    // Собираем обновленный вектор b
    MPI_Gather(
        localb.data(), rows_per_proc, MPI_DOUBLE, 
        rank == 0 ? b.data() : nullptr, rows_per_proc, MPI_DOUBLE, 
        0, MPI_COMM_WORLD
    );

    // Засекаем время окончания вычислений
    double t_end = MPI_Wtime();

    if (rank == 0) {
        // ОБРАТНЫЙ ХОД (выполняется последовательно на главном процессе)
        // 
        std::vector<double> x(N); // Вектор неизвестных
        for (int i = N - 1; i >= 0; --i) {
            double sum = 0.0;
            // Находим сумму уже вычисленных корней
            for (int j = i + 1; j < N; ++j) {
                sum += A[i * N + j] * x[j];
            }
            // Вычисляем значение переменной x_i
            x[i] = (b[i] - sum) / A[i * N + i];
        }

        // Вывод итогов
        std::cout << "=== Task 2: Gaussian Elimination (N=" << N << ", P=" << size << ") ===\n";
        std::cout << "Execution time (forward MPI-part + gather): " << (t_end - t_start) << " seconds\n";
        std::cout << "Solution x (first min(10,N) entries):\n";
        int limit = std::min(N, 10); // Ограничиваем вывод, если матрица огромная
        for (int i = 0; i < limit; ++i) {
            std::cout << "x[" << i << "] = " << x[i] << "\n";
        }
    }

    // Завершаем работу MPI, освобождаем системные ресурсы
    MPI_Finalize(); 
    return 0;
}