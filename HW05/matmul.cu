// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW05.pdf

#include "matmul.cuh"

// Tiled matrix multiplication kernel using shared memory.
// Each block computes a block_dim x block_dim tile of C.
// Shared memory holds one tile of A and one tile of B at a time.
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
    // Dynamic shared memory: first half for tile_A, second half for tile_B
    extern __shared__ char smem[];
    T *tile_A = reinterpret_cast<T *>(smem);
    T *tile_B = tile_A + blockDim.x * blockDim.y;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int bsize = blockDim.x; // blockDim.x == blockDim.y == block_dim

    T sum = static_cast<T>(0);
    const int num_tiles = (n + bsize - 1) / bsize;

    for (int t = 0; t < num_tiles; t++) {
        // Column of A tile and row of B tile for this iteration
        const int a_col = t * bsize + tx;
        const int b_row = t * bsize + ty;

        // Load tile of A (pad with 0 for out-of-bounds)
        tile_A[ty * bsize + tx] =
            (row < (int)n && a_col < (int)n) ? A[row * n + a_col] : static_cast<T>(0);

        // Load tile of B (pad with 0 for out-of-bounds)
        tile_B[ty * bsize + tx] =
            (b_row < (int)n && col < (int)n) ? B[b_row * n + col] : static_cast<T>(0);

        __syncthreads();

        // Accumulate partial dot product for this tile
        for (int k = 0; k < bsize; k++) {
            sum += tile_A[ty * bsize + k] * tile_B[k * bsize + tx];
        }

        __syncthreads();
    }

    if (row < (int)n && col < (int)n) {
        C[row * n + col] = sum;
    }
}

template <typename T>
static void matmul_impl(const T *A, const T *B, T *C, unsigned int n,
                        unsigned int block_dim) {
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    // Two tiles of block_dim x block_dim elements
    size_t smem = 2ULL * block_dim * block_dim * sizeof(T);
    matmul_kernel<T><<<grid, block, smem>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim) {
    matmul_impl<int>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim) {
    matmul_impl<float>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim) {
    matmul_impl<double>(A, B, C, n, block_dim);
}
