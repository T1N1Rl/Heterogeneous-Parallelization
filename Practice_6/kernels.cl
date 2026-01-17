// kernels.cl

__kernel void vector_add(
    __global const float* A,
    __global const float* B,
    __global float* C)
{
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}

__kernel void matrix_mul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int N, int M, int K)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if(row < N && col < K) {
        float sum = 0.0f;
        for(int i = 0; i < M; i++)
            sum += A[row * M + i] * B[i * K + col];
        C[row * K + col] = sum;
    }
}
