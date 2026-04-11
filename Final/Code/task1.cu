// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — Task 1: serial vs OpenMP vs GPU throughput benchmark
//
// Usage: ./task1 n t [tpb]
//   n    number of hashes to compute per algorithm
//   t    OpenMP thread count for parallel CPU run
//   tpb  CUDA threads per block (default: 256)
//
// Output (12 lines):
//   <md5_check_byte>       last byte of MD5(input_0) — correctness reference
//   <md5_serial_ms>        wall time for n serial MD5 hashes
//   <md5_omp_ms>           wall time for n OpenMP MD5 hashes (t threads)
//   <md5_gpu_ms>           kernel time for n GPU MD5 hashes
//   <sha1_check_byte>
//   <sha1_serial_ms>
//   <sha1_omp_ms>
//   <sha1_gpu_ms>
//   <sha256_check_byte>
//   <sha256_serial_ms>
//   <sha256_omp_ms>
//   <sha256_gpu_ms>
//
// Each hash i receives uint64_t i encoded as 8-byte little-endian input.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>

#include "md5_cpu.h"
#include "sha1_cpu.h"
#include "sha256_cpu.h"
#include "md5.cuh"
#include "sha1.cuh"
#include "sha256.cuh"

static inline void idx_to_msg(uint64_t i, uint8_t msg[8]) {
    for (int b = 0; b < 8; b++)
        msg[b] = (uint8_t)(i >> (8 * b));
}

// ── CPU serial ────────────────────────────────────────────────────────────────

template<typename HashFn>
static double cpu_serial(size_t n, HashFn fn, size_t digest_len) {
    uint8_t msg[8], digest[32];
    (void)digest_len;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; i++) {
        idx_to_msg((uint64_t)i, msg);
        fn(msg, 8, digest);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── CPU OpenMP ────────────────────────────────────────────────────────────────

template<typename HashFn>
static double cpu_omp(size_t n, int t, HashFn fn, size_t digest_len) {
    (void)digest_len;
    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(t) schedule(static)
    for (size_t i = 0; i < n; i++) {
        uint8_t msg[8], digest[32];
        idx_to_msg((uint64_t)i, msg);
        fn(msg, 8, digest);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── GPU ───────────────────────────────────────────────────────────────────────

template<typename GpuFn>
static float gpu_time(uint64_t n, int tpb, size_t digest_bytes, GpuFn fn) {
    uint8_t *d_digests;
    cudaMalloc(&d_digests, n * digest_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    fn(n, d_digests, tpb);   // calls cudaDeviceSynchronize() internally

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_digests);
    return ms;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " n t [tpb]\n"
                  << "  n    hashes per algorithm\n"
                  << "  t    OpenMP threads\n"
                  << "  tpb  CUDA threads per block (default 256)\n";
        return 1;
    }
    const size_t   n   = static_cast<size_t>(std::stoull(argv[1]));
    const int      t   = static_cast<int>(std::stoi(argv[2]));
    const int      tpb = (argc == 4) ? std::stoi(argv[3]) : 256;

    // Compute check bytes on CPU (last byte of hash(0), independent of n)
    uint8_t msg0[8] = {};
    uint8_t d_md5[16], d_sha1[20], d_sha256[32];
    md5_cpu(msg0, 8, d_md5);
    sha1_cpu(msg0, 8, d_sha1);
    sha256_cpu(msg0, 8, d_sha256);

    // ── MD5 ──
    double  md5_s  = cpu_serial(n, md5_cpu,    16);
    double  md5_o  = cpu_omp   (n, t, md5_cpu, 16);
    float   md5_g  = gpu_time  ((uint64_t)n, tpb, 16, md5_gpu);
    std::cout << (unsigned)d_md5[15] << "\n"
              << md5_s  << "\n"
              << md5_o  << "\n"
              << md5_g  << "\n";

    // ── SHA-1 ──
    double  sha1_s = cpu_serial(n, sha1_cpu,    20);
    double  sha1_o = cpu_omp   (n, t, sha1_cpu, 20);
    float   sha1_g = gpu_time  ((uint64_t)n, tpb, 20, sha1_gpu);
    std::cout << (unsigned)d_sha1[19] << "\n"
              << sha1_s  << "\n"
              << sha1_o  << "\n"
              << sha1_g  << "\n";

    // ── SHA-256 ──
    double  sha256_s = cpu_serial(n, sha256_cpu,    32);
    double  sha256_o = cpu_omp   (n, t, sha256_cpu, 32);
    float   sha256_g = gpu_time  ((uint64_t)n, tpb, 32, sha256_gpu);
    std::cout << (unsigned)d_sha256[31] << "\n"
              << sha256_s  << "\n"
              << sha256_o  << "\n"
              << sha256_g  << "\n";

    return 0;
}
