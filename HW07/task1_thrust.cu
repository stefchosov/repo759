// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW07.pdf

// HW07 Task 1 (Thrust): reduction using thrust::reduce.
// Usage: ./task1_thrust n
//
// Output (2 lines):
//   result   -- sum of all elements
//   ms       -- time for thrust::reduce in milliseconds

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[1]);

    // Create and fill host vector with random floats in [-1, 1]
    thrust::host_vector<float> h_vec(n);
    std::mt19937 gen(759);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        h_vec[i] = dist(gen);
    }

    // Copy to device using Thrust built-in
    thrust::device_vector<float> d_vec = h_vec;

    // Time thrust::reduce
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float result = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << result << "\n";
    std::cout << ms << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
