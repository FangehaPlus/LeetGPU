#include <cuda_runtime.h>

__device__ __forceinline__ void device_swap(float& x, float& y) {
    float t = x;
    x = y;
    y = t;
}

__global__ void reverse_array(float* input, int N) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x < N / 2) {
        device_swap(input[x], input[N - x - 1]);
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
