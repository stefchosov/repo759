#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include "matmul.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return 1;
    }

    const unsigned int n = static_cast<unsigned int>(std::atoi(argv[1]));
    const size_t size = static_cast<size_t>(n) * n;

    double *A = new double[size];
    double *B = new double[size];
    double *C = new double[size]();

    std::mt19937 gen(759);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < size; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << ms << "\n";
    std::cout << C[0] << "\n";
    std::cout << C[size - 1] << "\n";

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
