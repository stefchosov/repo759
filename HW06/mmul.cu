// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW06.pdf

#include "mmul.h"

// Uses cublasSgemm to compute C := A * B + C for n x n column-major matrices.
// alpha = 1.0 and beta = 1.0 so the result is C := 1.0 * A * B + 1.0 * C.
// A, B, C are all stored in column-major order (cuBLAS convention).
void mmul(cublasHandle_t handle, const float *A, const float *B, float *C, int n) {
    const float alpha = 1.0f;
    const float beta  = 1.0f;

    // cublasSgemm: C := alpha * op(A) * op(B) + beta * C
    // CUBLAS_OP_N means no transpose.
    // Leading dimensions equal n for square column-major matrices.
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                A, n,
                B, n,
                &beta,
                C, n);

    // Synchronize for accurate timing in the caller
    cudaDeviceSynchronize();
}
