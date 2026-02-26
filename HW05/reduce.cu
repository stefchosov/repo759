// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW05.pdf

#include "reduce.cuh"

// Parallel reduction kernel using "first add during global load" (Kernel 4).
// Each block processes 2 * blockDim.x elements: each thread loads one element
// at index i and immediately adds the element at index i + blockDim.x during
// the global load phase, halving the active thread count from the start.
// The partial block sum is written to g_odata[blockIdx.x].
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // First add during global load
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write this block's partial sum to output
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Repeatedly calls reduce_kernel until the full sum of the original N-element
// array is obtained. The final sum is written to (*input)[0].
// *input and *output are device arrays; no computation happens on the host.
__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block) {
    unsigned int n = N;
    float *cur_in = *input;
    float *cur_out = *output;
    const unsigned int smem = threads_per_block * sizeof(float);

    while (n > 1) {
        unsigned int num_blocks =
            (n + 2 * threads_per_block - 1) / (2 * threads_per_block);
        reduce_kernel<<<num_blocks, threads_per_block, smem>>>(cur_in, cur_out, n);

        // Swap buffers for the next round
        float *tmp = cur_in;
        cur_in = cur_out;
        cur_out = tmp;

        n = num_blocks;
    }

    // Ensure the final result lives in (*input)[0]
    if (cur_in != *input) {
        cudaMemcpy(*input, cur_in, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
}
