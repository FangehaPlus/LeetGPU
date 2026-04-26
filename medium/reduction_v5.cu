#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

/*
Reduction.
Unroll the for loop.
*/

__global__ void reduction(const float* input, float* output, int N) {
    volatile __shared__ float sram[THREADS_PER_BLOCK];
    /*
    `volatile` tells the compiler to access shared memory directly instead of optimizing accesses via registers.
    */
    int glob_offset = blockDim.x * blockIdx.x * 2;
    int idx_sram = threadIdx.x;
    sram[idx_sram] = (glob_offset + idx_sram) < N ? input[glob_offset + idx_sram] : 0.0f;
    sram[idx_sram] += (glob_offset + idx_sram + blockDim.x) < N ? input[glob_offset + idx_sram + blockDim.x] : 0.0f;
    __syncthreads();

    // for (int i = blockDim.x / 2; i > 32; i >>= 1) {
    //     if (idx_sram < i) {
    //         sram[idx_sram] += sram[idx_sram + i];
    //     }
    //     __syncthreads();
    // }

    if (THREADS_PER_BLOCK >= 512) {
        if (idx_sram < 256) {
            sram[idx_sram] += sram[idx_sram + 256];
        }
         __syncthreads();
    }

    if (THREADS_PER_BLOCK >= 256) {
        if (idx_sram < 128) {
            sram[idx_sram] += sram[idx_sram + 128];
        }
         __syncthreads();
    }

    if (THREADS_PER_BLOCK >= 128) {
        if (idx_sram < 64) {
            sram[idx_sram] += sram[idx_sram + 64];
        }
         __syncthreads();
    }

    if (idx_sram < 32) {
        sram[idx_sram] += sram[idx_sram + 32];
        sram[idx_sram] += sram[idx_sram + 16];
        sram[idx_sram] += sram[idx_sram + 8];
        sram[idx_sram] += sram[idx_sram + 4];
        sram[idx_sram] += sram[idx_sram + 2];
        sram[idx_sram] += sram[idx_sram + 1];
    }

    if (idx_sram == 0) {
        atomicAdd(output, sram[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 threadsPerBlock(THREADS_PER_BLOCK);

    // it should be ceil[N / (THREADS_PER_BLOCK * 2)]
    dim3 blocksPerGrid((N + threadsPerBlock.x * 2 - 1) / threadsPerBlock.x * 2);

    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    
}

