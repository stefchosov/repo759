// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
// Usage: Implementation based on problem specification from HW03.pdf

#include <iostream>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>
#include "vscale.cuh"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return 1;
    }

    // Read n from command line
    unsigned int n = std::atoi(argv[1]);

    // Create random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_a(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_b(0.0f, 1.0f);

    // Allocate and fill host arrays
    float *h_a = new float[n];
    float *h_b = new float[n];

    for (unsigned int i = 0; i < n; i++) {
        h_a[i] = dist_a(gen);
        h_b[i] = dist_b(gen);
    }

    // Allocate device arrays
    float *d_a, *d_b;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Calculate grid dimensions (512 threads per block)
    const int threadsPerBlock = 512;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Record start event
    cudaEventRecord(start);

    // Launch vscale kernel
    vscale<<<numBlocks, threadsPerBlock>>>(d_a, d_b, n);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print timing
    std::cout << milliseconds << std::endl;

    // Print first element
    std::cout << h_b[0] << std::endl;

    // Print last element
    std::cout << h_b[n - 1] << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
