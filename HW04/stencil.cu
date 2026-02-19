// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
// Usage: Implementation based on problem specification from HW04.pdf

#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output,
                                unsigned int n, unsigned int R) {
    // Shared memory layout:
    // - mask: (2*R + 1) elements
    // - image_s: blockDim.x + 2*R elements (includes halo)
    // - output_s: blockDim.x elements
    extern __shared__ float shared_mem[];

    float* mask_s = shared_mem;
    float* image_s = &mask_s[2 * R + 1];
    float* output_s = &image_s[blockDim.x + 2 * R];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load mask into shared memory (cooperatively)
    unsigned int mask_size = 2 * R + 1;
    for (unsigned int i = tid; i < mask_size; i += blockDim.x) {
        mask_s[i] = mask[i];
    }

    // Load image elements into shared memory with halos
    // Each block needs blockDim.x + 2*R elements from the image
    int block_start = blockIdx.x * blockDim.x - R;

    // Load main element for this thread
    if (gid < n) {
        image_s[tid + R] = image[gid];
    } else {
        image_s[tid + R] = 1.0f;  // Out of bounds
    }

    // Load left halo (first R threads load left elements)
    if (tid < R) {
        int img_idx = block_start + tid;
        if (img_idx >= 0 && img_idx < (int)n) {
            image_s[tid] = image[img_idx];
        } else {
            image_s[tid] = 1.0f;  // Out of bounds
        }
    }

    // Load right halo (last R threads load right elements)
    if (tid < R) {
        int img_idx = block_start + blockDim.x + R + tid;
        if (img_idx >= 0 && img_idx < (int)n) {
            image_s[blockDim.x + R + tid] = image[img_idx];
        } else {
            image_s[blockDim.x + R + tid] = 1.0f;  // Out of bounds
        }
    }

    __syncthreads();

    // Compute convolution for this thread
    if (gid < n) {
        float sum = 0.0f;
        for (int j = -((int)R); j <= (int)R; j++) {
            sum += image_s[tid + R + j] * mask_s[j + R];
        }
        output_s[tid] = sum;
    }

    __syncthreads();

    // Write result back to global memory
    if (gid < n) {
        output[gid] = output_s[tid];
    }
}

void stencil(const float* image, const float* mask, float* output,
             unsigned int n, unsigned int R, unsigned int threads_per_block) {
    // Calculate number of blocks
    unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    // Calculate shared memory size needed
    // mask: 2*R + 1, image: threads_per_block + 2*R, output: threads_per_block
    size_t shared_mem_size = ((2 * R + 1) + (threads_per_block + 2 * R) + threads_per_block) * sizeof(float);

    // Launch kernel with dynamic shared memory
    stencil_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(image, mask, output, n, R);
}
