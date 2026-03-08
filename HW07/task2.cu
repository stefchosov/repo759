// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW07.pdf

// HW07 Task 2: test harness for count().
// Usage: ./task2 n
//
// Output (3 lines):
//   values.back()   -- last (largest) unique integer
//   counts.back()   -- its occurrence count
//   ms              -- time for count() in milliseconds

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "count.cuh"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[1]);

    // Create host vector with random ints in [0, 500]
    thrust::host_vector<int> h_vec(n);
    std::mt19937 gen(759);
    std::uniform_int_distribution<int> dist(0, 500);
    for (int i = 0; i < n; i++) {
        h_vec[i] = dist(gen);
    }

    // Copy to device
    thrust::device_vector<int> d_in = h_vec;

    // Output vectors
    thrust::device_vector<int> values, counts;

    // Time the count call
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    count(d_in, values, counts);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << values.back() << "\n";
    std::cout << counts.back() << "\n";
    std::cout << ms << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
