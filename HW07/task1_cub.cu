// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW07.pdf

// HW07 Task 1 (CUB): reduction using cub::DeviceReduce::Sum.
// Usage: ./task1_cub n
//
// Output (2 lines):
//   result   -- sum of all elements
//   ms       -- time for DeviceReduce::Sum in milliseconds

#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <random>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[1]);

    // Create and fill host array with random floats in [-1, 1]
    float *h_in = new float[n];
    std::mt19937 gen(759);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        h_in[i] = dist(gen);
    }

    // Allocate device input and output, copy from host
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // Query temporary storage size
    void  *d_temp  = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
    cudaMalloc(&d_temp, temp_bytes);

    // Time the actual reduction (not the sizing call)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back to host
    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << result << "\n";
    std::cout << ms << "\n";

    delete[] h_in;
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
