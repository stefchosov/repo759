#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "matmul.h"

/*
Credit with support from ChatGPT in code generation:

Prompt: Using the specification provided and the header file, generate the code to meet the needs of task3.
*/

int main()
{
    const unsigned int n = 1024; // >= 1000 as required :contentReference[oaicite:2]{index=2}

    std::vector<double> Avec(n * n, 1.0);
    std::vector<double> Bvec(n * n, 1.0);

    // For mmul1/2/3 we need raw pointers
    const double *A = Avec.data();
    const double *B = Bvec.data();

    double *C = new double[n * n];

    auto run = [&](auto fn)
    {
        std::fill(C, C + (static_cast<std::size_t>(n) * n), 0.0);

        auto s = std::chrono::high_resolution_clock::now();
        fn();
        auto e = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(e - s).count();
        std::cout << ms << "\n";
        std::cout << C[static_cast<std::size_t>(n) * n - 1] << "\n";
    };

    std::cout << n << "\n";

    run([&]
        { mmul1(A, B, C, n); });
    run([&]
        { mmul2(A, B, C, n); });
    run([&]
        { mmul3(A, B, C, n); });
    run([&]
        { mmul4(Avec, Bvec, C, n); });

    delete[] C;
}
