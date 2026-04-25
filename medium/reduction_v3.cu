#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

/*
Reduction.
Deal with idle threads.
There are two ways introduced by 
`https://www.bilibili.com/video/BV1HvBSY2EJW/?p=5&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=c543471480985b18f978d07bc6486fbd`
Here, we implement the first way.
*/

__global__ void reduction(const float* input, float* output, int N) {
    __shared__ float sram[THREADS_PER_BLOCK];
    int glob_offset = blockDim.x * blockIdx.x * 2;
    int idx_sram = threadIdx.x;
    sram[idx_sram] = (glob_offset + idx_sram) < N ? input[glob_offset + idx_sram] : 0.0f;
    sram[idx_sram] += (glob_offset + idx_sram + blockDim.x) < N ? input[glob_offset + idx_sram + blockDim.x] : 0.0f;
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

    // it should be ceil[N / (THREADS_PER_BLOCK * 2)]
    dim3 blocksPerGrid((N + threadsPerBlock.x * 2 - 1) / threadsPerBlock.x * 2);

    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    
}


