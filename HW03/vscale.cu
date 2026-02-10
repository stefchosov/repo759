#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n) {
    // Calculate global thread index
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (idx < n) {
        // Element-wise multiplication: b[i] = a[i] * b[i]
        b[idx] = a[idx] * b[idx];
    }
}
