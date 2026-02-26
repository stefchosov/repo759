// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW05.pdf

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "matmul.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n block_dim" << std::endl;
        return 1;
    }

    const unsigned int n = static_cast<unsigned int>(std::atoi(argv[1]));
    const unsigned int block_dim = static_cast<unsigned int>(std::atoi(argv[2]));
    const size_t size = static_cast<size_t>(n) * n;

    std::mt19937 gen(759);

    // --- matmul_1: int ---
    {
        std::uniform_int_distribution<int> dist(1, 10);

        int *A, *B, *C;
        cudaMallocManaged(&A, size * sizeof(int));
        cudaMallocManaged(&B, size * sizeof(int));
        cudaMallocManaged(&C, size * sizeof(int));

        for (size_t i = 0; i < size; i++) {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }

        // Prefetch to GPU before timing
        int device;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(A, size * sizeof(int), device);
        cudaMemPrefetchAsync(B, size * sizeof(int), device);
        cudaMemPrefetchAsync(C, size * sizeof(int), device);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        matmul_1(A, B, C, n, block_dim);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << C[0] << "\n";
        std::cout << C[size - 1] << "\n";
        std::cout << ms << "\n";

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // --- matmul_2: float ---
    {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        float *A, *B, *C;
        cudaMallocManaged(&A, size * sizeof(float));
        cudaMallocManaged(&B, size * sizeof(float));
        cudaMallocManaged(&C, size * sizeof(float));

        for (size_t i = 0; i < size; i++) {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }

        int device;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(A, size * sizeof(float), device);
        cudaMemPrefetchAsync(B, size * sizeof(float), device);
        cudaMemPrefetchAsync(C, size * sizeof(float), device);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        matmul_2(A, B, C, n, block_dim);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << C[0] << "\n";
        std::cout << C[size - 1] << "\n";
        std::cout << ms << "\n";

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // --- matmul_3: double ---
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        double *A, *B, *C;
        cudaMallocManaged(&A, size * sizeof(double));
        cudaMallocManaged(&B, size * sizeof(double));
        cudaMallocManaged(&C, size * sizeof(double));

        for (size_t i = 0; i < size; i++) {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }

        int device;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(A, size * sizeof(double), device);
        cudaMemPrefetchAsync(B, size * sizeof(double), device);
        cudaMemPrefetchAsync(C, size * sizeof(double), device);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        matmul_3(A, B, C, n, block_dim);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << C[0] << "\n";
        std::cout << C[size - 1] << "\n";
        std::cout << ms << "\n";

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
