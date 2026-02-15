#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x * 4 + 3 < width * height * 4) {
        image[x * 4] = 255 - image[x * 4];
        image[x * 4 + 1] = 255 - image[x * 4 + 1];
        image[x * 4 + 2] = 255 - image[x * 4 + 2];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}

