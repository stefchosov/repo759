// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — Task 1: serial vs OpenMP CPU throughput benchmark
//
// Usage: ./task1 n t
//   n  — number of hashes to compute per algorithm
//   t  — number of OpenMP threads for the parallel run
//
// Output (9 lines):
//   <md5_check_byte>      last byte of MD5(input_0) — same for serial and OMP
//   <md5_serial_ms>       wall time for n serial MD5 hashes
//   <md5_omp_ms>          wall time for n parallel MD5 hashes (t threads)
//   <sha1_check_byte>
//   <sha1_serial_ms>
//   <sha1_omp_ms>
//   <sha256_check_byte>
//   <sha256_serial_ms>
//   <sha256_omp_ms>
//
// Each hash i receives an 8-byte little-endian encoding of uint64_t i as input.
// Hash computations are independent — OMP parallelism is embarrassingly parallel.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>

#include "md5_cpu.h"
#include "sha1_cpu.h"
#include "sha256_cpu.h"

// Encode index i as 8-byte little-endian message
static inline void idx_to_msg(uint64_t i, uint8_t msg[8]) {
    for (int b = 0; b < 8; b++)
        msg[b] = (uint8_t)(i >> (8 * b));
}

// ---- MD5 ---------------------------------------------------------------

static double run_md5_serial(size_t n) {
    uint8_t msg[8], digest[16];
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; i++) {
        idx_to_msg((uint64_t)i, msg);
        md5_cpu(msg, 8, digest);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_md5_omp(size_t n, int t) {
    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(t) schedule(static)
    for (size_t i = 0; i < n; i++) {
        uint8_t msg[8], digest[16];
        idx_to_msg((uint64_t)i, msg);
        md5_cpu(msg, 8, digest);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---- SHA-1 -------------------------------------------------------------

static double run_sha1_serial(size_t n) {
    uint8_t msg[8], digest[20];
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; i++) {
        idx_to_msg((uint64_t)i, msg);
        sha1_cpu(msg, 8, digest);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_sha1_omp(size_t n, int t) {
    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(t) schedule(static)
    for (size_t i = 0; i < n; i++) {
        uint8_t msg[8], digest[20];
        idx_to_msg((uint64_t)i, msg);
        sha1_cpu(msg, 8, digest);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---- SHA-256 -----------------------------------------------------------

static double run_sha256_serial(size_t n) {
    uint8_t msg[8], digest[32];
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; i++) {
        idx_to_msg((uint64_t)i, msg);
        sha256_cpu(msg, 8, digest);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_sha256_omp(size_t n, int t) {
    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(t) schedule(static)
    for (size_t i = 0; i < n; i++) {
        uint8_t msg[8], digest[32];
        idx_to_msg((uint64_t)i, msg);
        sha256_cpu(msg, 8, digest);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---- main --------------------------------------------------------------

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n t\n"
                  << "  n  number of hashes per algorithm\n"
                  << "  t  OpenMP thread count\n";
        return 1;
    }

    const size_t n = static_cast<size_t>(std::stoull(argv[1]));
    const int    t = static_cast<int>(std::stoi(argv[2]));

    // Compute check byte: last byte of each hash of input 0
    uint8_t msg0[8] = {};   // input 0 = all zeros (little-endian 0)
    uint8_t d_md5[16], d_sha1[20], d_sha256[32];
    md5_cpu(msg0, 8, d_md5);
    sha1_cpu(msg0, 8, d_sha1);
    sha256_cpu(msg0, 8, d_sha256);

    // --- MD5 ---
    double md5_serial = run_md5_serial(n);
    double md5_omp    = run_md5_omp(n, t);
    std::cout << (unsigned)d_md5[15]   << "\n"
              << md5_serial            << "\n"
              << md5_omp               << "\n";

    // --- SHA-1 ---
    double sha1_serial = run_sha1_serial(n);
    double sha1_omp    = run_sha1_omp(n, t);
    std::cout << (unsigned)d_sha1[19]  << "\n"
              << sha1_serial           << "\n"
              << sha1_omp              << "\n";

    // --- SHA-256 ---
    double sha256_serial = run_sha256_serial(n);
    double sha256_omp    = run_sha256_omp(n, t);
    std::cout << (unsigned)d_sha256[31] << "\n"
              << sha256_serial          << "\n"
              << sha256_omp             << "\n";

    return 0;
}
