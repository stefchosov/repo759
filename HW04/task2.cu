// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
// Usage: Implementation based on problem specification from HW04.pdf

#include <iostream>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>
#include "stencil.cuh"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n R threads_per_block" << std::endl;
        return 1;
    }

    // Parse command line arguments
    unsigned int n = std::atoi(argv[1]);
    unsigned int R = std::atoi(argv[2]);
    unsigned int threads_per_block = std::atoi(argv[3]);

    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Allocate and fill host arrays
    float* h_image = new float[n];
    float* h_output = new float[n];
    unsigned int mask_size = 2 * R + 1;
    float* h_mask = new float[mask_size];

    for (unsigned int i = 0; i < n; i++) {
        h_image[i] = dist(gen);
    }

    for (unsigned int i = 0; i < mask_size; i++) {
        h_mask[i] = dist(gen);
    }

    // Allocate device memory
    float *d_image, *d_output, *d_mask;
    cudaMalloc((void**)&d_image, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));
    cudaMalloc((void**)&d_mask, mask_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_image, h_image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Call stencil function
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print last element of output array
    std::cout << h_output[n - 1] << std::endl;

    // Print timing
    std::cout << milliseconds << std::endl;

    // Cleanup
    delete[] h_image;
    delete[] h_output;
    delete[] h_mask;
    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_mask);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
