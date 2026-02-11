// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
// Usage: Implementation based on problem specification from HW03.pdf

#include <iostream>
#include <cuda_runtime.h>

__global__ void factorial_kernel(int *dA) {
    int idx = threadIdx.x;
    int result = 1;

    // Compute (idx + 1)!
    for (int i = 1; i <= idx + 1; i++) {
        result *= i;
    }

    dA[idx] = result;
}

int main() {
    const int n = 8;
    int hA[n];
    int *dA;

    // Allocate device memory
    cudaMalloc((void**)&dA, n * sizeof(int));

    // Launch kernel with 1 block and 8 threads
    factorial_kernel<<<1, 8>>>(dA);

    // Copy results back to host
    cudaMemcpy(hA, dA, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results (one per line)
    for (int i = 0; i < n; i++) {
        std::cout << hA[i] << std::endl;
    }

    // Free device memory
    cudaFree(dA);

    return 0;
}
