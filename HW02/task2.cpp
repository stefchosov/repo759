#include <iostream>
#include <random>
#include <chrono>
#include <cstddef>
#include "convolution.h"

/*
Credit with support from ChatGPT in code generation:

Prompt: Using the specification provided and the header file, generate the code to meet the needs of task2.
*/

int main(int argc, char **argv)
{
    std::size_t n = static_cast<std::size_t>(std::stoull(argv[1]));
    std::size_t m = static_cast<std::size_t>(std::stoull(argv[2]));

    float *image = new float[n * n];
    float *mask = new float[m * m];
    float *output = new float[n * n];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> di(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dm(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n * n; ++i)
        image[i] = di(gen);
    for (std::size_t i = 0; i < m * m; ++i)
        mask[i] = dm(gen);

    auto start = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << ms << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[n * n - 1] << "\n";

    delete[] image;
    delete[] mask;
    delete[] output;
}
