// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW08.pdf

#include "matmul.h"

// Parallel version of mmul2 from HW02 (i-k-j loop order).
// Thread count is controlled by the caller via omp_set_num_threads().
// The i-k-j ordering keeps B accesses sequential (cache-friendly) while
// allowing the outer i loop to be parallelised without data races.
void mmul(const float *A, const float *B, float *C, const std::size_t n) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t k = 0; k < n; k++) {
            for (std::size_t j = 0; j < n; j++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
