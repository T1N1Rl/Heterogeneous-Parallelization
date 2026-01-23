#include <mpi.h>      
#include <iostream>   
#include <vector>     
#include <random>    
#include <iomanip>    
#include <cmath>      


// Задание 3: Параллельный алгоритм Флойда–Уоршелла
int main(int argc, char** argv) {
    // Инициализация MPI окружения
    MPI_Init(&argc, &argv); 

    int rank = 0, size = 1;
    // Получаем номер текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    // Получаем общее количество запущенных процессов
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int N = 8; // Размер матрицы смежности графа по умолчанию
    // Если передан аргумент командной строки, обновляем N
    if (argc >= 2) {
        N = std::atoi(argv[1]); 
        if (N <= 0) N = 8;
    }

    // Определяем константу "Бесконечность" для отсутствующих ребер
    const double INF = 1e12;

    std::vector<double> dist_full;  // Вектор для хранения полной матрицы (только для rank 0)
    
    // Подготовка графа на главном процессе
    if (rank == 0) {
        dist_full.resize(N * N); // Выделяем память под всю матрицу N x N
        std::mt19937 gen(42);    // Инициализируем генератор случайных чисел
        std::uniform_int_distribution<int> wdist(1, 9); // Веса ребер от 1 до 9
        std::uniform_real_distribution<double> prob(0.0, 1.0); // Вероятность наличия ребра

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == j) {
                    dist_full[i * N + j] = 0.0; // Расстояние до самого себя всегда 0
                } else {
                    // С вероятностью 50% создаем ребро, иначе — пути нет (INF)
                    if (prob(gen) < 0.5) {
                        dist_full[i * N + j] = static_cast<double>(wdist(gen));
                    } else {
                        dist_full[i * N + j] = INF;
                    }
                }
            }
        }
    }

    // РАСЧЕТ РАСПРЕДЕЛЕНИЯ СТРОК (логика как в предыдущих задачах)
    std::vector<int> rows_counts(size), row_displs(size);
    int base = N / size; // Сколько строк минимум получит каждый процесс
    int rem  = N % size;  // Остаток строк для распределения между первыми процессами
    int row_offset = 0;
    
    for (int r = 0; r < size; ++r) {
        int rows = base + (r < rem ? 1 : 0); // Добавляем по одной строке из остатка
        rows_counts[r] = rows;
        row_displs[r]  = row_offset;
        row_offset    += rows;
    }

    // Преобразуем количество строк в количество элементов (умножаем на длину строки N)
    std::vector<int> countsElems(size), displsElems(size);
    for (int r = 0; r < size; ++r) {
        countsElems[r] = rows_counts[r] * N;
        displsElems[r] = row_displs[r] * N;
    }

    int local_rows = rows_counts[rank]; // Количество строк текущего процесса
    int start_row  = row_displs[rank]; // Глобальный индекс первой строки этого процесса

    // Буфер для хранения локальной части матрицы
    std::vector<double> local_block(local_rows * N);

    // 
    // Раздаем части матрицы от rank 0 всем остальным процессам
    MPI_Scatterv(
        rank == 0 ? dist_full.data() : nullptr, // Исходные данные
        countsElems.data(), displsElems.data(), MPI_DOUBLE, // Описание частей
        local_block.data(), local_rows * N, MPI_DOUBLE, // Приемник
        0, MPI_COMM_WORLD // Корень и коммуникатор
    );

    // Основная рабочая матрица на каждом процессе (здесь будет храниться полная копия)
    std::vector<double> dist(N * N);

    // Первоначальный обмен: каждый процесс отдает свой блок и получает блоки других
    // 
    MPI_Allgatherv(
        local_block.data(), local_rows * N, MPI_DOUBLE, // Мой кусок
        dist.data(), countsElems.data(), displsElems.data(), MPI_DOUBLE, // Куда собираем все
        MPI_COMM_WORLD
    );

    // Синхронизация перед началом замеров времени
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // АЛГОРИТМ ФЛОЙДА-УОРШЕЛЛА
    // k — промежуточная вершина, через которую пытаемся улучшить путь
    for (int k = 0; k < N; ++k) {
        // Каждый процесс обновляет только те пути i -> j, где i — его локальная строка
        for (int i_local = 0; i_local < local_rows; ++i_local) {
            int i = start_row + i_local; // Глобальный индекс строки

            double dik = dist[i * N + k]; // Расстояние от i до промежуточной вершины k
            if (dik >= INF) continue;     // Если до k не дойти, пропускаем

            for (int j = 0; j < N; ++j) {
                // Пытаемся улучшить путь i -> j через вершину k
                double alt = dik + dist[k * N + j]; 
                if (alt < dist[i * N + j]) {
                    dist[i * N + j] = alt; // Нашли путь короче!
                }
            }
        }

        // СИНХРОНИЗАЦИЯ: После каждой итерации k процессы должны обменяться результатами.
        // Без этого процессам на шаге k+1 будут недоступны сокращенные пути, найденные соседями.
        MPI_Allgatherv(
            dist.data() + start_row * N, // Отправляем наш обновленный локальный блок
            local_rows * N, MPI_DOUBLE, 
            dist.data(), countsElems.data(), displsElems.data(), MPI_DOUBLE, // Обновляем всю матрицу dist
            MPI_COMM_WORLD
        );
    }

    // Замеряем время окончания вычислений
    double t_end = MPI_Wtime();

    // Вывод результатов только на процессе 0
    if (rank == 0) {
        std::cout << "=== Task 3: Floyd-Warshall (N=" << N << ", P=" << size << ") ===\n";
        std::cout << "Execution time: " << (t_end - t_start) << " seconds\n";

        // Если граф небольшой, выводим его матрицу расстояний на экран
        if (N <= 10) {
            std::cout << "Result distance matrix:\n";
            std::cout << std::fixed << std::setprecision(1); // Один знак после запятой
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    double val = dist[i * N + j];
                    if (val >= INF / 2) std::cout << "INF "; // Если пути нет
                    else std::cout << val << " "; // Если путь найден
                }
                std::cout << "\n";
            }
        } else {
            std::cout << "Matrix too large to print (N > 10).\n";
        }
    }

    // Освобождение ресурсов MPI
    MPI_Finalize(); 
    return 0;
}