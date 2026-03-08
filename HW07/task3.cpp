// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW07.pdf

// HW07 Task 3: OpenMP intro — thread introductions + parallel factorial.
// Compile: g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

#include <iostream>
#include <omp.h>

int main() {
    omp_set_num_threads(4);

    // Print thread count once; each thread introduces itself
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of threads: " << omp_get_num_threads() << "\n";

        #pragma omp critical
        std::cout << "I am thread No. " << omp_get_thread_num() << "\n";
    }

    // Compute and print factorials of 1..8 in parallel
    #pragma omp parallel for num_threads(4) schedule(static, 2)
    for (int i = 1; i <= 8; i++) {
        long long fact = 1;
        for (int j = 1; j <= i; j++) fact *= j;
        #pragma omp critical
        std::cout << i << "!=" << fact << "\n";
    }

    return 0;
}
