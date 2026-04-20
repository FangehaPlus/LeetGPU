#include <cuda_fp16.h>
#include <cuda_runtime.h>


__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= M || y >= N) return;
    float sum = beta * __half2float(C[x * N + y]);
    for (int k = 0; k < K; k++) {
        sum += alpha * __half2float(A[x * K + k]) * __half2float(B[k * N + y]);
    }
    C[x * N + y] = __float2half(sum);
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 block_dim(16, 16);
    dim3 grid_dim((M + 15) / 16, (N + 15) / 16);
    gemm_kernel<<<grid_dim, block_dim>>>(A, B, C, M, N, K, alpha, beta);
}
