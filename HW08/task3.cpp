// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW08.pdf

// HW08 Task 3: OpenMP parallel merge sort.
// Usage: ./task3 n t ts
//   n  -- array length
//   t  -- number of OpenMP threads
//   ts -- threshold: sub-arrays <= ts elements are sorted serially
//
// Output (3 lines):
//   arr[0]     -- first element after sort (smallest)
//   arr[n - 1] -- last element after sort (largest)
//   ms         -- time for msort in milliseconds

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <random>

#include "msort.h"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n t ts" << std::endl;
        return 1;
    }

    const std::size_t n  = static_cast<std::size_t>(std::atol(argv[1]));
    const int t          = std::atoi(argv[2]);
    const std::size_t ts = static_cast<std::size_t>(std::atoi(argv[3]));

    int *arr = new int[n];

    std::mt19937 gen(759);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (std::size_t i = 0; i < n; i++) arr[i] = dist(gen);

    omp_set_num_threads(t);

    auto t0 = std::chrono::high_resolution_clock::now();
    msort(arr, n, ts);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << arr[0]     << "\n";
    std::cout << arr[n - 1] << "\n";
    std::cout << ms         << "\n";

    delete[] arr;

    return 0;
}
