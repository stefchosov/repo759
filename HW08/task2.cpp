// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW08.pdf

// HW08 Task 2: OpenMP parallel 2D convolution.
// Usage: ./task2 n t
//   n -- image dimension (n x n)
//   t -- number of OpenMP threads
//
// Output (3 lines):
//   output[0]       -- first element of result
//   output[n*n - 1] -- last element of result
//   ms              -- time for convolve in milliseconds

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <random>

#include "convolution.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n t" << std::endl;
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::atoi(argv[1]));
    const int t = std::atoi(argv[2]);
    const std::size_t m = 3;  // 3x3 mask per spec

    float *image  = new float[n * n];
    float *mask   = new float[m * m];
    float *output = new float[n * n]();

    std::mt19937 gen(759);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < n * n; i++) image[i] = dist(gen);
    for (std::size_t i = 0; i < m * m; i++) mask[i]  = dist(gen);

    omp_set_num_threads(t);

    auto t0 = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << output[0]         << "\n";
    std::cout << output[n * n - 1] << "\n";
    std::cout << ms                << "\n";

    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}
