#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

/*
Reduction.
Bank Conflict
*/

__global__ void reduction(const float* input, float* output, int N) {
    __shared__ float sram[THREADS_PER_BLOCK];
    int idx_glob = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_sram = threadIdx.x;
    sram[idx_sram] = idx_glob < N ? input[idx_glob] : 0.0f;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (idx_sram < i) {
            sram[idx_sram] += sram[idx_sram + i];
        }
        __syncthreads();
    }
    if (idx_sram == 0) {
        atomicAdd(output, sram[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    
}


