// lab9.cpp
// Практическая работа №9: MPI — распределённая обработка данных
//
// Запуск:
//   mpic++ lab9.cpp -O2 -o lab9
//   mpirun -np 4 ./lab9 1 1000000   // Задание 1, N=1e6
//   mpirun -np 4 ./lab9 2 8         // Задание 2, N=8
//   mpirun -np 4 ./lab9 3 8         // Задание 3, N=8

#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

using namespace std;

// ------------------------------------------------------------
// Вспомогалка для красивого вывода usage
// ------------------------------------------------------------
void print_usage(int rank) {
    if (rank == 0) {
        cerr << "Usage: mpirun -np <P> ./lab9 <task> [N]\n"
             << "  task = 1 : distributed mean & stddev (N default = 1000000)\n"
             << "  task = 2 : distributed Gaussian elimination (N default = 8)\n"
             << "  task = 3 : Floyd-Warshall on graph (N default = 8)\n";
    }
}

// ------------------------------------------------------------
// ЗАДАНИЕ 1: Среднее и стандартное отклонение (MPI_Scatterv + MPI_Reduce)
// ------------------------------------------------------------
void task1_mean_std(int N, int rank, int size) {
    int world_rank = rank;
    int world_size = size;

    vector<double> data;
    if (world_rank == 0) {
        data.resize(N);
        mt19937 gen(42);
        uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < N; ++i) {
            data[i] = dist(gen);
        }
    }

    // Распределяем элементы с учётом остатка: MPI_Scatterv
    vector<int> sendcounts(world_size), displs(world_size);
    int base = N / world_size;
    int rem  = N % world_size;

    int offset = 0;
    for (int r = 0; r < world_size; ++r) {
        int cnt = base + (r < rem ? 1 : 0);
        sendcounts[r] = cnt;
        displs[r] = offset;
        offset += cnt;
    }

    int local_n = sendcounts[world_rank];
    vector<double> local_data(local_n);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    MPI_Scatterv(
        world_rank == 0 ? data.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        local_data.data(),
        local_n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Локальная статистика
    double local_sum = 0.0;
    double local_sum_sq = 0.0;
    for (int i = 0; i < local_n; ++i) {
        double x = local_data[i];
        local_sum    += x;
        local_sum_sq += x * x;
    }

    double global_sum = 0.0;
    double global_sum_sq = 0.0;

    MPI_Reduce(&local_sum,    &global_sum,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();

    if (world_rank == 0) {
        double mean = global_sum / static_cast<double>(N);
        double var  = global_sum_sq / static_cast<double>(N) - mean * mean;
        if (var < 0.0) var = 0.0;
        double stddev = sqrt(var);

        cout << "=== Task 1: Mean & Std (N=" << N << ", P=" << world_size << ") ===\n";
        cout << "Mean     = " << mean << "\n";
        cout << "Std dev  = " << stddev << "\n";
        cout << "Exec time= " << (t_end - t_start) << " seconds\n";
    }
}

// ------------------------------------------------------------
// ЗАДАНИЕ 2: Распределённое решение СЛАУ методом Гаусса
//   - строки A и b делим между процессами (Scatterv)
//   - прямой ход — параллельно, pivot-строка раздаётся через MPI_Bcast
//   - обратный ход — на rank=0 после Gatherv
// ------------------------------------------------------------
void task2_gauss(int N, int rank, int size) {
    int world_rank = rank;
    int world_size = size;

    vector<double> A; // NxN
    vector<double> b; // N

    // Распределение строк между процессами
    vector<int> rows_counts(world_size), row_displs(world_size);
    int base = N / world_size;
    int rem  = N % world_size;
    int row_offset = 0;
    for (int r = 0; r < world_size; ++r) {
        int rows = base + (r < rem ? 1 : 0);
        rows_counts[r] = rows;
        row_displs[r]  = row_offset;
        row_offset    += rows;
    }

    // Для Scatterv/Gatherv по элементам
    vector<int> sendcountsA(world_size), displsA(world_size);
    for (int r = 0; r < world_size; ++r) {
        sendcountsA[r] = rows_counts[r] * N;
        displsA[r]     = row_displs[r] * N;
    }

    if (world_rank == 0) {
        A.resize(N * N);
        b.resize(N);

        // Генерируем диагонально доминирующую матрицу для стабильности
        mt19937 gen(42);
        uniform_real_distribution<double> dist(1.0, 10.0);

        for (int i = 0; i < N; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                double val = dist(gen);
                A[i * N + j] = val;
                row_sum += fabs(val);
            }
            A[i * N + i] = row_sum + 1.0; // доминирование
            b[i] = dist(gen);
        }
    }

    int local_rows = rows_counts[world_rank];
    vector<double> localA(local_rows * N);
    vector<double> localb(local_rows);

    // Рассылаем строки A и b
    MPI_Scatterv(
        world_rank == 0 ? A.data() : nullptr,
        sendcountsA.data(),
        displsA.data(),
        MPI_DOUBLE,
        localA.data(),
        local_rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Scatterv(
        world_rank == 0 ? b.data() : nullptr,
        rows_counts.data(),
        row_displs.data(),
        MPI_DOUBLE,
        localb.data(),
        local_rows,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // На всех процессах нужен маппинг "глобальная строка -> владелец"
    vector<int> owner_row(N);
    for (int r = 0; r < world_size; ++r) {
        int start = row_displs[r];
        int cnt   = rows_counts[r];
        for (int i = 0; i < cnt; ++i) {
            owner_row[start + i] = r;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // ---- Прямой ход Гаусса (без выбора главного элемента) ----
    for (int k = 0; k < N; ++k) {
        int owner = owner_row[k];

        vector<double> pivotRow(N);
        double pivotB = 0.0;

        if (world_rank == owner) {
            int local_k = k - row_displs[world_rank];
            double* row_ptr = &localA[local_k * N];

            for (int j = 0; j < N; ++j) {
                pivotRow[j] = row_ptr[j];
            }
            pivotB = localb[local_k];
        }

        // Распространяем pivot строку и правую часть
        MPI_Bcast(pivotRow.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivotB, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        double pivot = pivotRow[k];
        if (fabs(pivot) < 1e-12) {
            // Без выбора главного элемента просто пропускаем, но в норме
            // матрица должна быть корректной (диагонально доминирующей).
            continue;
        }

        // Обновляем свои строки
        for (int i_local = 0; i_local < local_rows; ++i_local) {
            int i_global = row_displs[world_rank] + i_local;
            if (i_global <= k) continue;

            double a_ik = localA[i_local * N + k];
            if (fabs(a_ik) < 1e-12) continue;

            double factor = a_ik / pivot;
            for (int j = k; j < N; ++j) {
                localA[i_local * N + j] -= factor * pivotRow[j];
            }
            localb[i_local] -= factor * pivotB;
        }
    }

    // Собираем треугольную матрицу и модифицированный вектор b на rank 0
    if (world_rank == 0) {
        A.assign(N * N, 0.0);
        b.assign(N, 0.0);
    }

    MPI_Gatherv(
        localA.data(),
        local_rows * N,
        MPI_DOUBLE,
        world_rank == 0 ? A.data() : nullptr,
        sendcountsA.data(),
        displsA.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gatherv(
        localb.data(),
        local_rows,
        MPI_DOUBLE,
        world_rank == 0 ? b.data() : nullptr,
        rows_counts.data(),
        row_displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double t_end = MPI_Wtime();

    if (world_rank == 0) {
        // ---- Обратный ход (back substitution) ----
        vector<double> x(N);
        for (int i = N - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < N; ++j) {
                sum += A[i * N + j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i * N + i];
        }

        cout << "=== Task 2: Gaussian Elimination (N=" << N << ", P=" << world_size << ") ===\n";
        cout << "Execution time (forward MPI-part only): " << (t_end - t_start) << " seconds\n";
        cout << "Solution x (first min(10,N) entries):\n";
        int limit = std::min(N, 10);
        for (int i = 0; i < limit; ++i) {
            cout << "x[" << i << "] = " << x[i] << "\n";
        }
    }
}

// ------------------------------------------------------------
// ЗАДАНИЕ 3: Параллельный Флойд–Уоршелл (MPI_Allgatherv)
//   - каждый процесс обновляет свой блок строк
//   - после каждой итерации k синхронизируемся через MPI_Allgatherv
// ------------------------------------------------------------
void task3_floyd(int N, int rank, int size) {
    int world_rank = rank;
    int world_size = size;

    const double INF = 1e12;

    vector<double> dist(N * N);

    if (world_rank == 0) {
        // Генерируем граф: случайные ребра, веса 1..9, часть отсутсвует -> INF
        mt19937 gen(42);
        uniform_int_distribution<int> wdist(1, 9);
        uniform_real_distribution<double> prob(0.0, 1.0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == j) {
                    dist[i * N + j] = 0.0;
                } else {
                    if (prob(gen) < 0.5) { // ~50% плотность графа
                        dist[i * N + j] = static_cast<double>(wdist(gen));
                    } else {
                        dist[i * N + j] = INF;
                    }
                }
            }
        }
    }

    // Рассылаем матрицу всем
    MPI_Bcast(dist.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Разбиваем строки между процессами
    vector<int> rows_counts(world_size), row_displs(world_size);
    int base = N / world_size;
    int rem  = N % world_size;
    int row_offset = 0;
    for (int r = 0; r < world_size; ++r) {
        int rows = base + (r < rem ? 1 : 0);
        rows_counts[r] = rows;
        row_displs[r]  = row_offset;
        row_offset    += rows;
    }

    vector<int> countsElems(world_size), displsElems(world_size);
    for (int r = 0; r < world_size; ++r) {
        countsElems[r] = rows_counts[r] * N;
        displsElems[r] = row_displs[r] * N;
    }

    int local_rows = rows_counts[world_rank];
    int start_row  = row_displs[world_rank];
    double* local_block = dist.data() + start_row * N;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Алгоритм Флойда–Уоршелла с Allgatherv после каждой итерации
    for (int k = 0; k < N; ++k) {
        for (int i_local = 0; i_local < local_rows; ++i_local) {
            int i = start_row + i_local;

            double dik = dist[i * N + k];
            if (dik >= INF) continue;

            for (int j = 0; j < N; ++j) {
                double alt = dik + dist[k * N + j];
                if (alt < dist[i * N + j]) {
                    dist[i * N + j] = alt;
                }
            }
        }

        // Синхронизируем обновленные строки всех процессов
        MPI_Allgatherv(
            local_block,
            local_rows * N,
            MPI_DOUBLE,
            dist.data(),
            countsElems.data(),
            displsElems.data(),
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );
    }

    double t_end = MPI_Wtime();

    if (world_rank == 0) {
        cout << "=== Task 3: Floyd-Warshall (N=" << N << ", P=" << world_size << ") ===\n";
        cout << "Execution time: " << (t_end - t_start) << " seconds\n";

        if (N <= 10) {
            cout << "Result distance matrix:\n";
            cout << fixed << setprecision(1);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    double val = dist[i * N + j];
                    if (val >= INF / 2) cout << "INF ";
                    else cout << val << " ";
                }
                cout << "\n";
            }
        } else {
            cout << "Matrix too large to print (N > 10).\n";
        }
    }
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        print_usage(rank);
        MPI_Finalize();
        return 0;
    }

    int task = std::atoi(argv[1]);
    int N = 0;

    if (task == 1) {
        N = (argc >= 3) ? std::atoi(argv[2]) : 1000000;
        if (N <= 0) N = 1000000;
        task1_mean_std(N, rank, size);
    } else if (task == 2) {
        N = (argc >= 3) ? std::atoi(argv[2]) : 8;
        if (N <= 0) N = 8;
        task2_gauss(N, rank, size);
    } else if (task == 3) {
        N = (argc >= 3) ? std::atoi(argv[2]) : 8;
        if (N <= 0) N = 8;
        task3_floyd(N, rank, size);
    } else {
        print_usage(rank);
    }

    MPI_Finalize();
    return 0;
}
