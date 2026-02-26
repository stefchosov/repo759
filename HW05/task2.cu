// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW05.pdf

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "reduce.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N threads_per_block" << std::endl;
        return 1;
    }

    const unsigned int N = static_cast<unsigned int>(std::atoi(argv[1]));
    const unsigned int threads_per_block =
        static_cast<unsigned int>(std::atoi(argv[2]));

    // Create host array and fill with random values in [-1, 1]
    float *h_data = new float[N];
    std::mt19937 gen(759);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (unsigned int i = 0; i < N; i++) {
        h_data[i] = dist(gen);
    }

    // Allocate device input array and copy from host
    float *d_input;
    cudaMalloc(&d_input, static_cast<size_t>(N) * sizeof(float));
    cudaMemcpy(d_input, h_data, static_cast<size_t>(N) * sizeof(float),
               cudaMemcpyHostToDevice);
    delete[] h_data;

    // Allocate device output array sized for the first kernel call
    const unsigned int num_blocks_first =
        (N + 2 * threads_per_block - 1) / (2 * threads_per_block);
    float *d_output;
    cudaMalloc(&d_output, static_cast<size_t>(num_blocks_first) * sizeof(float));

    // Time the reduce function
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce(&d_input, &d_output, N, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy the result (now in d_input[0]) back to host
    float result = 0.0f;
    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << result << "\n";
    std::cout << ms << "\n";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
