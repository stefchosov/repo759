// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW06.pdf

// Test harness for HW06 Task 2: Hillis-Steele inclusive scan.
// Usage: ./task2 n threads_per_block
//   n                 -- number of elements to scan (n <= tpb * tpb)
//   threads_per_block -- threads per block for the hillis_steele kernel
//
// Output (2 lines):
//   output[n - 1]  -- last element of the inclusive prefix sum
//   ms             -- elapsed time in milliseconds

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "scan.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n threads_per_block" << std::endl;
        return 1;
    }

    const unsigned int n = static_cast<unsigned int>(std::atoi(argv[1]));
    const unsigned int threads_per_block =
        static_cast<unsigned int>(std::atoi(argv[2]));

    // input and output must be managed memory (per scan.cuh contract)
    float *input, *output;
    cudaMallocManaged(&input,  n * sizeof(float));
    cudaMallocManaged(&output, n * sizeof(float));

    // Fill input with uniform random floats in [-1, 1]
    std::mt19937 gen(759);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (unsigned int i = 0; i < n; i++) {
        input[i] = dist(gen);
    }

    // Time the scan call
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    scan(input, output, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Print last prefix-sum element then timing
    std::cout << output[n - 1] << "\n";
    std::cout << ms << "\n";

    cudaFree(input);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
