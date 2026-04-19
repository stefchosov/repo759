// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — Task 3: truncated-collision search (CPU serial + OpenMP + GPU)
//
// GPU algorithm: Pollard's rho with distinguished points.
//   Each thread walks an independent chain x_{n+1} = truncate(hash(x_n)).
//   A "distinguished point" (DP) is any x with its low dp_bits bits zero.
//   Two chains that reach the same DP value from different starts imply a
//   collision in the truncated hash function (birthday paradox in the cycle).
//   Memory: O(1) per thread — scales to bits=64 without GPU memory limits.
//
// CPU algorithms: sequential and OpenMP-parallel unordered_map scan.
//   Phase 1 (parallel): threads hash disjoint index ranges into per-batch buffers.
//   Phase 2 (serial): merge into shared map, detect first duplicate.
//   Run for bits <= CPU_MAX_BITS; larger sizes reported as -1.
//
// Truncated hash: first 8 digest bytes as LE uint64, masked to low 'bits' bits.
//   MD5 (LE output):  bytes 0-7 as LE uint64 = st[0] | (st[1] << 32)
//   SHA (BE output):  bytes 0-7 as LE uint64 = bswap32(st[0]) | (bswap32(st[1]) << 32)
//
// Usage: ./task3                    — all combos (3 algos x 7 bits)
//        ./task3 <algo>             — one algo, all 7 bits  (e.g. md5)
//        ./task3 <algo> <bits>      — single combo           (e.g. sha1 32)
//
// Output columns: algo  bits  cpu_count  cpu_ms  omp_count  omp_ms  gpu_count  gpu_ms  expected
//   cpu_count / cpu_ms / omp_count / omp_ms = -1 when bits > CPU_MAX_BITS

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>

#include "md5_cpu.h"
#include "sha1_cpu.h"
#include "sha256_cpu.h"

static constexpr int CPU_MAX_BITS = 56;

// ── GPU constant memory ──────────────────────────────────────────────────────

__constant__ static uint32_t t3_md5_K[64] = {
    0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu,
    0xf57c0fafu, 0x4787c62au, 0xa8304613u, 0xfd469501u,
    0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu,
    0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u,
    0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau,
    0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
    0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu,
    0xa9e3e905u, 0xfcefa3f8u, 0x676f02d9u, 0x8d2a4c8au,
    0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu,
    0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
    0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u,
    0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
    0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u,
    0x655b59c3u, 0x8f0ccc92u, 0xffeff47du, 0x85845dd1u,
    0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
    0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u
};
__constant__ static uint32_t t3_md5_S[64] = {
     7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
     5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
     4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
     6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21
};
__constant__ static uint32_t t3_sha256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

// ── Device helpers ────────────────────────────────────────────────────────────

__device__ static inline uint32_t rotl32d(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32u - n));
}
__device__ static inline uint32_t rotr32d(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}
__device__ static inline uint32_t bswap32d(uint32_t x) {
    return ((x & 0xffu) << 24) | ((x & 0xff00u) << 8)
         | ((x >> 8)  & 0xff00u) | ((x >> 24) & 0xffu);
}

// ── Per-algorithm device functions returning first 8 digest bytes as LE uint64 ──
// Matching the CPU extraction:
//   (uint64_t)d[0]|(d[1]<<8)|(d[2]<<16)|(d[3]<<24) | ((uint64_t)d[4..7] << 32)
// MD5 (LE output): bytes 0-7 = st[0],st[1] in LE  →  st[0] | (st[1] << 32)
// SHA (BE output): bytes 0-7 = bswap pairs         →  bswap32(st[0]) | (bswap32(st[1]) << 32)

__device__ static uint64_t md5_le64(uint64_t i) {
    uint32_t M[16] = {};
    M[0] = (uint32_t)(i); M[1] = (uint32_t)(i >> 32);
    M[2] = 0x00000080u;   M[14] = 0x00000040u;

    uint32_t a0=0x67452301u, b0=0xefcdab89u, c0=0x98badcfeu, d0=0x10325476u;
    uint32_t a=a0, b=b0, c=c0, d=d0;
    for (int j = 0; j < 64; j++) {
        uint32_t f, g;
        if      (j < 16) { f = (b & c) | (~b & d); g = (uint32_t)j;       }
        else if (j < 32) { f = (d & b) | (~d & c); g = (5u*j + 1u) % 16u; }
        else if (j < 48) { f = b ^ c ^ d;           g = (3u*j + 5u) % 16u; }
        else             { f = c ^ (b | ~d);         g = (7u*j)      % 16u; }
        f += a + t3_md5_K[j] + M[g];
        a = d; d = c; c = b; b += rotl32d(f, t3_md5_S[j]);
    }
    return (uint64_t)(a0 + a) | ((uint64_t)(b0 + b) << 32);
}

__device__ static uint64_t sha1_le64(uint64_t i) {
    uint32_t W[80] = {};
    W[0]  = bswap32d((uint32_t)(i));
    W[1]  = bswap32d((uint32_t)(i >> 32));
    W[2]  = 0x80000000u;
    W[15] = 0x00000040u;
    for (int j = 16; j < 80; j++)
        W[j] = rotl32d(W[j-3] ^ W[j-8] ^ W[j-14] ^ W[j-16], 1);

    uint32_t h0=0x67452301u, h1=0xefcdab89u;
    uint32_t a=h0, b=h1, c=0x98badcfeu, d=0x10325476u, e=0xc3d2e1f0u;
    for (int j = 0; j < 80; j++) {
        uint32_t f, k;
        if      (j < 20) { f = (b & c) | (~b & d);          k = 0x5a827999u; }
        else if (j < 40) { f = b ^ c ^ d;                   k = 0x6ed9eba1u; }
        else if (j < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdcu; }
        else             { f = b ^ c ^ d;                    k = 0xca62c1d6u; }
        uint32_t tmp = rotl32d(a, 5) + f + e + k + W[j];
        e = d; d = c; c = rotl32d(b, 30); b = a; a = tmp;
    }
    return (uint64_t)bswap32d(h0 + a) | ((uint64_t)bswap32d(h1 + b) << 32);
}

__device__ static uint64_t sha256_le64(uint64_t i) {
    uint32_t W[64] = {};
    W[0]  = bswap32d((uint32_t)(i));
    W[1]  = bswap32d((uint32_t)(i >> 32));
    W[2]  = 0x80000000u;
    W[15] = 0x00000040u;
    for (int j = 16; j < 64; j++) {
        uint32_t s0 = rotr32d(W[j-15],  7) ^ rotr32d(W[j-15], 18) ^ (W[j-15] >>  3);
        uint32_t s1 = rotr32d(W[j- 2], 17) ^ rotr32d(W[j- 2], 19) ^ (W[j- 2] >> 10);
        W[j] = W[j-16] + s0 + W[j-7] + s1;
    }
    uint32_t h0=0x6a09e667u, h1=0xbb67ae85u;
    uint32_t a=h0,  b=h1,  c=0x3c6ef372u, d=0xa54ff53au;
    uint32_t e=0x510e527fu, f=0x9b05688cu, g=0x1f83d9abu, h=0x5be0cd19u;
    for (int j = 0; j < 64; j++) {
        uint32_t S1  = rotr32d(e,  6) ^ rotr32d(e, 11) ^ rotr32d(e, 25);
        uint32_t ch  = (e & f) ^ (~e & g);
        uint32_t t1  = h + S1 + ch + t3_sha256_K[j] + W[j];
        uint32_t S0  = rotr32d(a,  2) ^ rotr32d(a, 13) ^ rotr32d(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2  = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    return (uint64_t)bswap32d(h0 + a) | ((uint64_t)bswap32d(h1 + b) << 32);
}

// ── Distinguished-point entry ─────────────────────────────────────────────────

struct DPEntry {
    uint64_t start;
    uint64_t dp_val;
};

// ── Pollard's rho kernel ──────────────────────────────────────────────────────
// Each thread walks chain x_{n+1} = hash_fn(x_n) & hash_mask.
// Reports (start, x) when x & dp_mask == 0 (distinguished point).
// Restarts from a fresh starting value after each DP or when chain is too long.
// Template<ALGO> ensures no warp divergence — compiler emits three specializations.

template<int ALGO>
__global__ void pollard_kernel(
    uint64_t hash_mask, uint64_t dp_mask,
    uint64_t chain_offset, uint64_t n_threads,
    DPEntry *dp_buf, unsigned long long *dp_cnt,
    uint64_t dp_cap, uint64_t iters_per_thread, uint64_t max_chain_len
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    uint64_t chain_idx = tid;
    uint64_t x         = (chain_offset + chain_idx) & hash_mask;
    uint64_t start     = x;
    uint64_t steps     = 0;

    for (uint64_t iter = 0; iter < iters_per_thread; iter++) {
        uint64_t h;
        if      constexpr (ALGO == 0) h = md5_le64(x);
        else if constexpr (ALGO == 1) h = sha1_le64(x);
        else                          h = sha256_le64(x);
        x = h & hash_mask;
        steps++;

        bool is_dp      = (x & dp_mask) == 0;
        bool chain_long = (steps >= max_chain_len);

        if (is_dp) {
            unsigned long long idx = atomicAdd(dp_cnt, 1ULL);
            if (idx < (unsigned long long)dp_cap) {
                dp_buf[idx].start  = start;
                dp_buf[idx].dp_val = x;
            }
        }

        if (is_dp || chain_long) {
            // Start fresh chain from the next unique slot for this thread
            chain_idx += n_threads;
            x      = (chain_offset + chain_idx) & hash_mask;
            start  = x;
            steps  = 0;
        }
    }
}

// ── GPU collision search (Pollard's rho) ──────────────────────────────────────

static void gpu_collision(int algo, int bits, float *ms_out, uint64_t *count_out) {
    // dp_bits = bits/4: expected chain length 2^dp_bits, expected DPs before
    //   collision = sqrt(pi/2) * 2^(bits/4) << dp_buf capacity.
    const int      dp_bits       = bits / 4;
    const uint64_t hash_mask     = (bits < 64) ? ((1ull << bits) - 1ull) : ~0ull;
    const uint64_t dp_mask       = (1ull << dp_bits) - 1ull;
    const uint64_t n_threads     = 1ull << 16;   // 65 536
    const int      tpb           = 256;
    const uint64_t iters_per_thr = 4ull << dp_bits;  // ~4 DPs per thread per launch
    const uint64_t max_chain_len = 16ull << dp_bits;  // restart after 16x expected
    const uint64_t dp_cap        = n_threads * 8;     // generous buffer

    DPEntry            *d_dp_buf = nullptr;
    unsigned long long *d_dp_cnt = nullptr;
    cudaMalloc(&d_dp_buf, dp_cap * sizeof(DPEntry));
    cudaMalloc(&d_dp_cnt, sizeof(unsigned long long));

    std::vector<DPEntry> h_buf(dp_cap);
    std::unordered_map<uint64_t, uint64_t> seen;
    seen.reserve(1u << std::min(dp_bits + 4, 22));

    uint64_t chain_offset = 0;
    bool     found        = false;
    int      blocks       = (int)(n_threads / tpb);
    *count_out            = 0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    while (!found) {
        cudaMemset(d_dp_cnt, 0, sizeof(unsigned long long));

        if      (algo == 0)
            pollard_kernel<0><<<blocks, tpb>>>(hash_mask, dp_mask, chain_offset,
                n_threads, d_dp_buf, d_dp_cnt, dp_cap, iters_per_thr, max_chain_len);
        else if (algo == 1)
            pollard_kernel<1><<<blocks, tpb>>>(hash_mask, dp_mask, chain_offset,
                n_threads, d_dp_buf, d_dp_cnt, dp_cap, iters_per_thr, max_chain_len);
        else
            pollard_kernel<2><<<blocks, tpb>>>(hash_mask, dp_mask, chain_offset,
                n_threads, d_dp_buf, d_dp_cnt, dp_cap, iters_per_thr, max_chain_len);

        cudaDeviceSynchronize();

        unsigned long long n_dps_raw = 0;
        cudaMemcpy(&n_dps_raw, d_dp_cnt, sizeof(n_dps_raw), cudaMemcpyDeviceToHost);
        uint64_t n_dps = std::min((uint64_t)n_dps_raw, dp_cap);
        cudaMemcpy(h_buf.data(), d_dp_buf, n_dps * sizeof(DPEntry), cudaMemcpyDeviceToHost);

        for (uint64_t k = 0; k < n_dps && !found; k++) {
            uint64_t dv = h_buf[k].dp_val;
            uint64_t st = h_buf[k].start;
            auto it = seen.find(dv);
            if (it != seen.end() && it->second != st) {
                found = true;
            } else {
                seen.emplace(dv, st);
            }
        }

        *count_out   += n_threads * iters_per_thr;
        chain_offset += n_threads << 3;
    }

    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(ms_out, ev0, ev1);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_dp_buf);
    cudaFree(d_dp_cnt);
}

// ── CPU collision search ──────────────────────────────────────────────────────

static uint64_t cpu_collision(int algo, int bits, double *ms_out) {
    uint64_t mask = (bits < 64) ? ((1ull << bits) - 1ull) : ~0ull;

    std::unordered_map<uint64_t, uint64_t> seen;
    seen.reserve(1u << std::min(bits / 2 + 3, 22));
    uint8_t msg[8], digest[32];

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t idx = 0; ; idx++) {
        for (int k = 0; k < 8; k++) msg[k] = (uint8_t)(idx >> (8 * k));
        if      (algo == 0) md5_cpu(msg, 8, digest);
        else if (algo == 1) sha1_cpu(msg, 8, digest);
        else                sha256_cpu(msg, 8, digest);

        // First 8 bytes as LE uint64 (matches GPU extraction convention)
        uint64_t h = (uint64_t)digest[0]          | ((uint64_t)digest[1] <<  8)
                   | ((uint64_t)digest[2] << 16)   | ((uint64_t)digest[3] << 24)
                   | ((uint64_t)digest[4] << 32)   | ((uint64_t)digest[5] << 40)
                   | ((uint64_t)digest[6] << 48)   | ((uint64_t)digest[7] << 56);
        h &= mask;

        if (seen.count(h)) {
            auto t1 = std::chrono::high_resolution_clock::now();
            *ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return idx + 1;
        }
        seen[h] = idx;
    }
}

// ── OpenMP parallel collision search ─────────────────────────────────────────
// Phase 1 (parallel): n_thr threads each hash their slice of [base, base+batch)
//   into pre-allocated hbuf / ibuf arrays — no synchronization needed.
// Phase 2 (serial): merge into global map; stop at first duplicate key.
// Batch size tracks ~1/8 of the expected collision distance to minimize overshoot.

static uint64_t omp_collision(int algo, int bits, double *ms_out) {
    uint64_t mask    = (bits < 64) ? ((1ull << bits) - 1ull) : ~0ull;
    int      n_thr   = omp_get_max_threads();

    uint64_t expected_cnt = (uint64_t)(sqrt(M_PI / 2.0) * pow(2.0, bits / 2.0));
    uint64_t per_thread   = std::max((uint64_t)1, expected_cnt / 8 / (uint64_t)n_thr);
    uint64_t batch        = per_thread * (uint64_t)n_thr;

    std::unordered_map<uint64_t, uint64_t> seen;
    seen.reserve(1u << std::min(bits / 2 + 3, 22));

    std::vector<uint64_t> hbuf(batch), ibuf(batch);

    auto t0 = std::chrono::high_resolution_clock::now();

    uint64_t base  = 0;
    bool     found = false;
    uint64_t result = 0;

    while (!found) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)batch; i++) {
            uint64_t idx = base + (uint64_t)i;
            uint8_t  msg[8], digest[32];
            for (int k = 0; k < 8; k++) msg[k] = (uint8_t)(idx >> (8 * k));
            if      (algo == 0) md5_cpu(msg, 8, digest);
            else if (algo == 1) sha1_cpu(msg, 8, digest);
            else                sha256_cpu(msg, 8, digest);
            hbuf[(size_t)i] = ((uint64_t)digest[0]          | ((uint64_t)digest[1] <<  8)
                             | ((uint64_t)digest[2] << 16)   | ((uint64_t)digest[3] << 24)
                             | ((uint64_t)digest[4] << 32)   | ((uint64_t)digest[5] << 40)
                             | ((uint64_t)digest[6] << 48)   | ((uint64_t)digest[7] << 56)) & mask;
            ibuf[(size_t)i] = idx;
        }

        for (uint64_t i = 0; i < batch && !found; i++) {
            auto res = seen.emplace(hbuf[i], ibuf[i]);
            if (!res.second) { found = true; result = ibuf[i] + 1; }
        }

        base += batch;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    *ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    static const char *algo_names[] = {"md5", "sha1", "sha256"};
    static const int   bits_arr[]   = {16, 24, 32, 40, 48, 56, 64};
    static const int   N_BITS       = 7;

    int a0 = 0, a1 = 3, b0 = 0, b1 = N_BITS;
    if (argc >= 2) {
        for (int a = 0; a < 3; a++)
            if (strcmp(argv[1], algo_names[a]) == 0) { a0 = a; a1 = a + 1; }
    }
    if (argc >= 3) {
        int bv = atoi(argv[2]);
        for (int bi = 0; bi < N_BITS; bi++)
            if (bits_arr[bi] == bv) { b0 = bi; b1 = bi + 1; }
    }

    printf("# algo  bits  cpu_count  cpu_ms  omp_count  omp_ms  gpu_count  gpu_ms  expected\n");
    fprintf(stderr, "OpenMP max_threads = %d\n", omp_get_max_threads());

    for (int a = a0; a < a1; a++) {
        for (int bi = b0; bi < b1; bi++) {
            int      bits     = bits_arr[bi];
            uint64_t expected = (uint64_t)(sqrt(M_PI / 2.0) * pow(2.0, bits / 2.0));

            int64_t  cpu_cnt = -1;
            double   cpu_ms  = -1.0;
            if (bits <= CPU_MAX_BITS) {
                cpu_cnt = (int64_t)cpu_collision(a, bits, &cpu_ms);
            }

            int64_t  omp_cnt = -1;
            double   omp_ms  = -1.0;
            if (bits <= CPU_MAX_BITS) {
                omp_cnt = (int64_t)omp_collision(a, bits, &omp_ms);
            }

            float    gpu_ms  = 0.0f;
            uint64_t gpu_cnt = 0;
            gpu_collision(a, bits, &gpu_ms, &gpu_cnt);

            printf("%s %d %lld %.3f %lld %.3f %llu %.3f %llu\n",
                   algo_names[a], bits,
                   (long long)cpu_cnt, cpu_ms,
                   (long long)omp_cnt, omp_ms,
                   (unsigned long long)gpu_cnt, gpu_ms,
                   (unsigned long long)expected);
            fflush(stdout);
        }
    }
    return 0;
}
