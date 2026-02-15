#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x < N) {
        output[x] = fmaxf(input[x], 0.0f);
    }
    /*
    max(a, b) / min(a, b)     # int, some types
    fmaxf(a, b) / fminf(a, b) # float
    fmax(a, b) / fmin(a, b)   # double
    */
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

