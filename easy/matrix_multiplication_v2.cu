#include <cuda_runtime.h>

#define L 64  // block tile size
#define S 16  // stride in Dimension for reduction
#define V 4  // thread tile size

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float a_sram[L][S], b_sram[S][L];
    float c_reg[V][V] = {0.0f};
    float a_reg[V], b_reg[V];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    int total_shared_elements = L * S;
    for (int n = 0; n < N; n += S) {
        __syncthreads();
        // cooperative fetching
        // fetch A
        for (int i = tid; i < total_shared_elements; i += num_threads) {
            int row_sram = i / S;
            int col_sram = i % S;
            int row_global = by * L + row_sram;
            int col_global = n + col_sram;
            if (row_global < M && col_global < N) {
                a_sram[row_sram][col_sram] = A[row_global * N + col_global];
            } else {
                a_sram[row_sram][col_sram] = 0.0f;
            }
        }
        // fetch B
        for (int i = tid; i < total_shared_elements; i += num_threads) {
            int row_sram = i / L;
            int col_sram = i % L;
            int row_global = n + row_sram;
            int col_global = bx * L + col_sram;
            if (row_global < N && col_global < K) {
                b_sram[row_sram][col_sram] = B[row_global * K + col_global];
            } else {
                b_sram[row_sram][col_sram] = 0.0f;
            }
        }
        __syncthreads();

        for (int s = 0; s < S; s++) {
            for (int i = 0; i < V; i++) {
                a_reg[i] = a_sram[ty * V + i][s];
                b_reg[i] = b_sram[s][tx * V + i];
            }
            for (int y = 0; y < V; y++) {
                for (int x = 0; x < V; x++) {
                    c_reg[y][x] += a_reg[y] * b_reg[x];
                }
            }
        }
    }
    for (int y = 0; y < V; y++) {
        for (int x = 0; x < V; x++) {
            int row_global = by * L + ty * V + y;
            int col_global = bx * L + tx * V + x;
            if (row_global < M && col_global < K) {
                C[row_global * K + col_global] = c_reg[y][x];
            }
        }
    }                                      
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(L / V, L / V);
    dim3 blocksPerGrid((K + L - 1) / L, (M + L - 1) / L);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
