// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
// Usage: Implementation based on problem specification from HW04.pdf

#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    // Calculate global thread index (1D configuration)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of elements in the output matrix
    size_t total = n * n;

    // Check bounds
    if (idx < total) {
        // Convert 1D index to 2D row and column
        size_t row = idx / n;
        size_t col = idx % n;

        // Compute dot product of row from A and column from B
        float sum = 0.0f;
        for (size_t k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }

        // Store result
        C[idx] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    // Total number of elements in output matrix
    size_t total = n * n;

    // Calculate number of blocks needed
    unsigned int num_blocks = (total + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);
}
