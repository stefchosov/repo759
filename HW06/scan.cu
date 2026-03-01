// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW06.pdf

#include "scan.cuh"

// Hillis-Steele inclusive prefix scan kernel.
// Each block independently scans its portion of the input array.
// Uses shared memory to perform O(n log n) work in O(log n) parallel steps.
// input:  global array, length n (or num_blocks for phase-2 block-sum scan)
// output: global array, length n
// n:      total number of valid elements (threads with gid >= n load 0)
__global__ void hillis_steele(const float *input, float *output, unsigned int n) {
    extern __shared__ float temp[];

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;

    // Load from global memory; pad out-of-bounds threads with 0
    temp[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // Log2(blockDim.x) scan steps (Hillis-Steele)
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        float val = (tid >= stride) ? temp[tid - stride] : 0.0f;
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write result; only valid global indices write to output
    if (gid < n) {
        output[gid] = temp[tid];
    }
}

// Adds the previous block's inclusive-scan sum to every element in the
// current block (for blocks 1, 2, ... num_blocks-1).
// scanned_block_sums[b] holds the inclusive sum of all elements in blocks 0..b,
// so we add scanned_block_sums[blockIdx.x - 1] to shift block b's local sums
// into global prefix sums.
__global__ void add_block_offsets(float *output,
                                  const float *scanned_block_sums,
                                  unsigned int n) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && gid < n) {
        output[gid] += scanned_block_sums[blockIdx.x - 1];
    }
}

// Performs an inclusive scan on managed-memory arrays input/output of length n.
// Assumption: n <= threads_per_block * threads_per_block
//
// Algorithm (3 phases):
//   1. Each block runs hillis_steele independently on its threads_per_block chunk.
//   2. The per-block sums are themselves scanned with a single-block hillis_steele.
//   3. add_block_offsets propagates the scanned block sums to the output array.
__host__ void scan(const float *input, float *output, unsigned int n,
                   unsigned int threads_per_block) {
    const unsigned int num_blocks =
        (n + threads_per_block - 1) / threads_per_block;
    const size_t smem = threads_per_block * sizeof(float);

    // Phase 1: local inclusive scan within each block
    hillis_steele<<<num_blocks, threads_per_block, smem>>>(input, output, n);
    cudaDeviceSynchronize();

    if (num_blocks == 1) {
        // Single block: the scan is already complete
        return;
    }

    // Phase 2: gather last element of each block's local scan (= block sum),
    // then compute the inclusive scan of those block sums.
    float *block_sums, *block_sums_scanned;
    cudaMallocManaged(&block_sums,         num_blocks * sizeof(float));
    cudaMallocManaged(&block_sums_scanned, num_blocks * sizeof(float));

    // Copy last element of each block's output into block_sums.
    // For block b, the last valid element index is min((b+1)*tpb, n) - 1.
    for (unsigned int b = 0; b < num_blocks; b++) {
        unsigned int last_idx = (b + 1) * threads_per_block;
        if (last_idx > n) last_idx = n;
        last_idx -= 1;
        cudaMemcpy(&block_sums[b], &output[last_idx],
                   sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();

    // Scan the block sums (at most threads_per_block sums, fits in one block)
    hillis_steele<<<1, threads_per_block, smem>>>(
        block_sums, block_sums_scanned, num_blocks);
    cudaDeviceSynchronize();

    // Phase 3: add scanned block sums as offsets to blocks 1..num_blocks-1
    add_block_offsets<<<num_blocks, threads_per_block>>>(
        output, block_sums_scanned, n);
    cudaDeviceSynchronize();

    cudaFree(block_sums);
    cudaFree(block_sums_scanned);
}
