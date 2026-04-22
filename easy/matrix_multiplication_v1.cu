#include <cuda_runtime.h>

/* NOTES:
 * 1. The X/Y to Column/Row Mapping Convention
 * - Rule: Always map the X dimension to Columns (K) and Y to Rows (M).
 * - Why: C/C++ uses row-major memory layout. Mapping X to columns ensures 
 * that consecutive threads in a Warp access contiguous memory addresses. 
 * This enables "Memory Coalescing", which is crucial for GPU performance.
 *
 * 2. Strict 2D Boundary Checking (Preventing "Row Spillover")
 * - Rule: Always use strict 2D coordinate checks: `if (row < M && col < K)`.
 * - Why: NEVER use a flat 1D capacity check like `if (idx < M * K)`. 
 * A 1D check fails to detect horizontal out-of-bounds threads that wrap 
 * around to the next row. For example, in a 3x3 matrix (capacity 9), a 
 * thread targeting (Row 0, Col 3) calculates a 1D index of 3. Since 3 < 9, 
 * it passes the check and silently overwrites valid data at (Row 1, Col 0).
 */

#define V 4

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float c[V][V] = {0};
    float a[V], b[V];
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < V; i++) {
            int idx_a_row = y * V + i;
            int idx_b_col = x * V + i;
            a[i] = idx_a_row < M ? A[idx_a_row * N + n] : 0.0f;
            b[i] = idx_b_col < K ? B[n * K + idx_b_col] : 0.0f;
        }
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                c[i][j] += a[i] * b[j];
            }
        }
    }
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            int row = y * V + i;
            int col = x * V + j;
            if (row < M && col < K) C[row * K + col] = c[i][j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    int elem_per_block_x = threadsPerBlock.x * V;
    int elem_per_block_y = threadsPerBlock.y * V;
    
    dim3 blocksPerGrid((K + elem_per_block_x - 1) / elem_per_block_x,
                       (M + elem_per_block_y - 1) / elem_per_block_y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
