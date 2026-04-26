#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

/*
Reduction.
use `__shfl_down_sync`
*/

__device__ __forceinline__ float wrap_reduce_sum(float sum) {
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

__global__ void reduction(const float* input, float* output, int N) {
    float sum = 0.f;
    int idx_glob = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    sum += idx_glob < N ? input[idx_glob] : 0.f;
    idx_glob += blockDim.x;
    sum += idx_glob < N ? input[idx_glob] : 0.f;

    sum = wrap_reduce_sum(sum);

    __syncthreads();

    __shared__ float wrapLevelSums[32];
    int wrap_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        wrapLevelSums[wrap_id] = wrap_id < THREADS_PER_BLOCK / 32 ? sum : 0.f;
    }

    __syncthreads();

    if (wrap_id == 0) {
        sum = wrapLevelSums[lane_id];
        sum = wrap_reduce_sum(sum);
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 threadsPerBlock(THREADS_PER_BLOCK);

    // it should be ceil[N / (THREADS_PER_BLOCK * 2)]
    dim3 blocksPerGrid((N + threadsPerBlock.x * 2 - 1) / threadsPerBlock.x * 2);

    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    
}


