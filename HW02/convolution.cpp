#include "convolution.h"

/*
Credit with support from ChatGPT in code generation:

Prompt: Using the specification provided and the header file, generate the code to meet the needs of convolution.cpp.
*/

static inline float padded(const float *img, std::size_t n, long i, long j)
{
    const long N = static_cast<long>(n);

    const bool in_i = (0 <= i && i < N);
    const bool in_j = (0 <= j && j < N);

    if (in_i && in_j)
    {
        return img[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(j)];
    }

    // corner: BOTH indices out of bounds
    if (!in_i && !in_j)
        return 0.0f;

    // edge (but not corner)
    return 1.0f;
}

void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m)
{
    const long r = static_cast<long>(m / 2);

    for (std::size_t x = 0; x < n; ++x)
    {
        for (std::size_t y = 0; y < n; ++y)
        {
            float sum = 0.0f;

            for (std::size_t i = 0; i < m; ++i)
            {
                for (std::size_t j = 0; j < m; ++j)
                {
                    long fx = static_cast<long>(x) + static_cast<long>(i) - r;
                    long fy = static_cast<long>(y) + static_cast<long>(j) - r;

                    sum += mask[i * m + j] * padded(image, n, fx, fy);
                }
            }

            output[x * n + y] = sum;
        }
    }
}
