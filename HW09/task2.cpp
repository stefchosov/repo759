// HW09 Task 2: Monte Carlo π estimation with OpenMP (with and without simd).
// Usage: ./task2 n t
//   n -- number of random 2D points
//   t -- number of OpenMP threads
//
// Output (2 lines):
//   pi_estimate  -- 4 * incircle / n
//   ms           -- time for the montecarlo function in milliseconds

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <random>

#include "montecarlo.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n t" << std::endl;
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::atol(argv[1]));
    const int t          = std::atoi(argv[2]);
    const float radius   = 1.0f;

    float *x = new float[n];
    float *y = new float[n];

    std::mt19937 gen(759);
    std::uniform_real_distribution<float> dist(-radius, radius);
    for (std::size_t i = 0; i < n; i++) {
        x[i] = dist(gen);
        y[i] = dist(gen);
    }

    omp_set_num_threads(t);

    auto t0 = std::chrono::high_resolution_clock::now();
    int incircle = montecarlo(n, x, y, radius);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float pi_est = 4.0f * static_cast<float>(incircle) / static_cast<float>(n);

    std::cout << pi_est << "\n";
    std::cout << ms     << "\n";

    delete[] x;
    delete[] y;

    return 0;
}
