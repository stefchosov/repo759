// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — Task 2: GPU thread-block scaling benchmark
//
// Usage: ./task2 n tpb
//   n    number of hashes per algorithm
//   tpb  CUDA threads per block
//
// Output (6 lines):
//   <md5_check_byte>
//   <md5_gpu_ms>
//   <sha1_check_byte>
//   <sha1_gpu_ms>
//   <sha256_check_byte>
//   <sha256_gpu_ms>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#include "md5_cpu.h"
#include "sha1_cpu.h"
#include "sha256_cpu.h"
#include "md5.cuh"
#include "sha1.cuh"
#include "sha256.cuh"

template<typename GpuFn>
static float gpu_time(uint64_t n, int tpb, size_t digest_bytes, GpuFn fn) {
    uint8_t *d_digests;
    cudaMalloc(&d_digests, n * digest_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    fn(n, d_digests, tpb);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_digests);
    return ms;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n tpb\n";
        return 1;
    }
    const uint64_t n   = std::stoull(argv[1]);
    const int      tpb = std::stoi(argv[2]);

    // Check bytes from CPU hash of input 0
    uint8_t msg0[8] = {};
    uint8_t d_md5[16], d_sha1[20], d_sha256[32];
    md5_cpu(msg0, 8, d_md5);
    sha1_cpu(msg0, 8, d_sha1);
    sha256_cpu(msg0, 8, d_sha256);

    float md5_ms   = gpu_time(n, tpb, 16, md5_gpu);
    std::cout << (unsigned)d_md5[15]    << "\n" << md5_ms   << "\n";

    float sha1_ms  = gpu_time(n, tpb, 20, sha1_gpu);
    std::cout << (unsigned)d_sha1[19]   << "\n" << sha1_ms  << "\n";

    float sha256_ms = gpu_time(n, tpb, 32, sha256_gpu);
    std::cout << (unsigned)d_sha256[31] << "\n" << sha256_ms << "\n";

    return 0;
}
