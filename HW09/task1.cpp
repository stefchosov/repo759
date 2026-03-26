// HW09 Task 1: OpenMP false-sharing study — cluster distance computation.
// Usage: ./task1 n t
//   n -- length of the arr array (must be a multiple of 2*t)
//   t -- number of OpenMP threads
//
// Output (3 lines):
//   max_dist      -- maximum distance across all partitions
//   partition_id  -- thread index that holds the maximum distance
//   ms            -- time for the cluster function in milliseconds

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "cluster.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n t" << std::endl;
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::atol(argv[1]));
    const std::size_t t = static_cast<std::size_t>(std::atoi(argv[2]));

    // Fill arr with random floats in [0, n], then sort.
    float *arr = new float[n];
    std::mt19937 gen(759);
    std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(n));
    for (std::size_t i = 0; i < n; i++) arr[i] = dist(gen);
    std::sort(arr, arr + n);

    // Centers: (2k+1)*n / (2t) for k = 0, 1, ..., t-1
    float *centers = new float[t];
    for (std::size_t k = 0; k < t; k++) {
        centers[k] = static_cast<float>(2 * k + 1) * static_cast<float>(n) /
                     static_cast<float>(2 * t);
    }

    float *dists = new float[t]();  // zero-initialised

    auto t0 = std::chrono::high_resolution_clock::now();
    cluster(n, t, arr, centers, dists);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Find partition with maximum distance.
    std::size_t max_id = 0;
    for (std::size_t i = 1; i < t; i++) {
        if (dists[i] > dists[max_id]) max_id = i;
    }

    std::cout << dists[max_id] << "\n";
    std::cout << max_id        << "\n";
    std::cout << ms            << "\n";

    delete[] arr;
    delete[] centers;
    delete[] dists;

    return 0;
}
