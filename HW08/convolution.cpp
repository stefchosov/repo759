// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW08.pdf

#include "convolution.h"

// Returns the image pixel value at (i, j) with zero-padding for out-of-bounds
// corners and 1-padding for out-of-bounds edges (matches HW02 spec).
static inline float padded(const float *img, std::size_t n, long i, long j) {
    const long N = static_cast<long>(n);
    const bool in_i = (i >= 0 && i < N);
    const bool in_j = (j >= 0 && j < N);
    if (in_i && in_j)  return img[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(j)];
    if (!in_i && !in_j) return 0.0f;
    return 1.0f;
}

// Parallel version of convolve from HW02.
// Thread count is controlled by the caller via omp_set_num_threads().
// Each output pixel is independent so the x-y loop nest is fully parallelisable.
void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) {
    const long r = static_cast<long>(m / 2);

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t x = 0; x < n; x++) {
        for (std::size_t y = 0; y < n; y++) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < m; j++) {
                    long fx = static_cast<long>(x) + static_cast<long>(i) - r;
                    long fy = static_cast<long>(y) + static_cast<long>(j) - r;
                    sum += mask[i * m + j] * padded(image, n, fx, fy);
                }
            }
            output[x * n + y] = sum;
        }
    }
}
