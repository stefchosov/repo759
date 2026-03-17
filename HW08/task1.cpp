// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW08.pdf

// HW08 Task 1: OpenMP parallel matrix multiplication.
// Usage: ./task1 n t
//   n -- matrix dimension
//   t -- number of OpenMP threads
//
// Output (3 lines):
//   C[0]       -- first element of result
//   C[n*n - 1] -- last element of result
//   ms         -- time for mmul in milliseconds

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <random>

#include "matmul.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n t" << std::endl;
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::atoi(argv[1]));
    const int t = std::atoi(argv[2]);
    const std::size_t size = n * n;

    float *A = new float[size];
    float *B = new float[size];
    float *C = new float[size]();  // zero-initialised

    std::mt19937 gen(759);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < size; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    omp_set_num_threads(t);

    auto t0 = std::chrono::high_resolution_clock::now();
    mmul(A, B, C, n);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << C[0]        << "\n";
    std::cout << C[size - 1] << "\n";
    std::cout << ms          << "\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
