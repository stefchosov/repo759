// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
// Usage: Implementation based on problem specification from HW04.pdf

#include <iostream>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>
#include "matmul.cuh"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n threads_per_block" << std::endl;
        return 1;
    }

    // Parse command line arguments
    size_t n = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Allocate and fill host matrices (row major)
    size_t size = n * n;
    float* h_A = new float[size];
    float* h_B = new float[size];
    float* h_C = new float[size];

    for (size_t i = 0; i < size; i++) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size * sizeof(float));
    cudaMalloc((void**)&d_B, size * sizeof(float));
    cudaMalloc((void**)&d_C, size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Call matmul function
    matmul(d_A, d_B, d_C, n, threads_per_block);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print last element of resulting matrix
    std::cout << h_C[size - 1] << std::endl;

    // Print timing
    std::cout << milliseconds << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
