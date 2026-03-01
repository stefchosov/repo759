// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW06.pdf

// Test harness for HW06 Task 1: cuBLAS matrix multiplication.
// Usage: ./task1 n
//   n           -- matrix dimension (A, B, C are n x n, column-major)
//
// Output (3 lines):
//   C[n*n - 1]  -- last element of the result matrix (column-major indexing)
//   ms          -- elapsed time in milliseconds (cuBLAS call + sync)

#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "mmul.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[1]);
    const size_t size = static_cast<size_t>(n) * n;

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate managed memory for column-major n x n matrices
    float *A, *B, *C;
    cudaMallocManaged(&A, size * sizeof(float));
    cudaMallocManaged(&B, size * sizeof(float));
    cudaMallocManaged(&C, size * sizeof(float));

    // Fill A and B with uniform random floats in [-1, 1]; C starts at zero
    std::mt19937 gen(759);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
        C[i] = 0.0f;
    }

    // Time the mmul call (includes cudaDeviceSynchronize inside mmul)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mmul(handle, A, B, C, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Print last element then timing
    std::cout << C[size - 1] << "\n";
    std::cout << ms << "\n";

    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
