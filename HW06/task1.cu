// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW06.pdf

// Test harness for HW06 Task 1: cuBLAS matrix multiplication.
// Usage: ./task1 n n_tests
//   n        -- matrix dimension (A, B, C are n x n, column-major)
//   n_tests  -- number of mmul calls; average time is reported
//
// Output (1 line):
//   avg_ms   -- average time per mmul call in milliseconds

#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "mmul.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n n_tests" << std::endl;
        return 1;
    }

    const int n       = std::atoi(argv[1]);
    const int n_tests = std::atoi(argv[2]);
    const size_t size = static_cast<size_t>(n) * n;

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate managed memory for column-major n x n matrices
    float *A, *B, *C;
    cudaMallocManaged(&A, size * sizeof(float));
    cudaMallocManaged(&B, size * sizeof(float));
    cudaMallocManaged(&C, size * sizeof(float));

    // Fill A, B, and C with uniform random floats in [-1, 1]
    std::mt19937 gen(759);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
        C[i] = dist(gen);
    }

    // Time n_tests calls to mmul; report average
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int t = 0; t < n_tests; t++) {
        mmul(handle, A, B, C, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << ms / n_tests << "\n";

    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
