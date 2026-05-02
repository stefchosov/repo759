// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — Task 4: GPU collision search via Thrust parallel sort
//
// Compares against Task 3's Pollard's rho. Three-phase algorithm:
//   Phase 1 (parallel kernel): hash batch_n inputs into device arrays
//   Phase 2 (Thrust):          thrust::sort_by_key on (hash, index) pairs
//   Phase 3 (parallel kernel): scan for first adjacent equal-hash pair
//
// Bit width is capped at 48 because the sort working set scales with
// 2^(bits/2) and exceeds 8 GB GPU VRAM at bits >= 52.
//
// Output columns: algo  bits  thrust_count  thrust_ms  expected
//
// Usage: ./task4
//        ./task4 <algo>
//        ./task4 <algo> <bits>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "gpu_hashes_narrow.cuh"

// ── Phase 1: hash batch into (hash, index) device arrays ─────────────────────

template<int ALGO>
__global__ void hash_to_arrays(
    uint64_t hash_mask, uint64_t base, uint64_t n,
    uint64_t *hashes, uint64_t *indices)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t idx = base + tid;
    uint64_t h;
    if      constexpr (ALGO == 0) h = gh_md5_le64(idx);
    else if constexpr (ALGO == 1) h = gh_sha1_le64(idx);
    else                          h = gh_sha256_le64(idx);
    hashes[tid]  = h & hash_mask;
    indices[tid] = idx;
}

// ── Phase 3: find first adjacent duplicate in sorted hash array ──────────────
// Reports max(idx[t], idx[t+1]) + 1 — the count of inputs evaluated up to
// and including the colliding one. atomicMin keeps the earliest such count.

__global__ void find_dup_kernel(
    const uint64_t *sorted_hashes, const uint64_t *sorted_indices,
    uint64_t n, unsigned long long *result_count)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid + 1 >= n) return;
    if (sorted_hashes[tid] == sorted_hashes[tid + 1]) {
        unsigned long long a = (unsigned long long)sorted_indices[tid];
        unsigned long long b = (unsigned long long)sorted_indices[tid + 1];
        unsigned long long cnt = (a > b ? a : b) + 1ULL;
        atomicMin(result_count, cnt);
    }
}

// ── Thrust collision search ──────────────────────────────────────────────────

static void thrust_collision(int algo, int bits, float *ms_out, uint64_t *count_out) {
    const uint64_t hash_mask = (bits < 64) ? ((1ull << bits) - 1ull) : ~0ull;
    const uint64_t expected  = (uint64_t)(sqrt(M_PI / 2.0) * pow(2.0, bits / 2.0));
    // 4× expected per batch: ~86.5% chance of finding collision per round.
    // Multi-round retry handles the misses.
    const uint64_t batch_n = std::max((uint64_t)4096, expected * 4);

    *count_out  = 0;
    bool found  = false;
    uint64_t base = 0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    // Allocate once, reuse across rounds
    thrust::device_vector<uint64_t>          d_hashes(batch_n);
    thrust::device_vector<uint64_t>          d_indices(batch_n);
    thrust::device_vector<unsigned long long> d_result(1);

    const int tpb         = 256;
    const int hash_blocks = (int)((batch_n + tpb - 1) / tpb);
    const int find_blocks = (int)((batch_n + tpb - 1) / tpb);

    while (!found) {
        // Phase 1: hash
        if      (algo == 0) hash_to_arrays<0><<<hash_blocks, tpb>>>(hash_mask, base, batch_n,
            thrust::raw_pointer_cast(d_hashes.data()),
            thrust::raw_pointer_cast(d_indices.data()));
        else if (algo == 1) hash_to_arrays<1><<<hash_blocks, tpb>>>(hash_mask, base, batch_n,
            thrust::raw_pointer_cast(d_hashes.data()),
            thrust::raw_pointer_cast(d_indices.data()));
        else                hash_to_arrays<2><<<hash_blocks, tpb>>>(hash_mask, base, batch_n,
            thrust::raw_pointer_cast(d_hashes.data()),
            thrust::raw_pointer_cast(d_indices.data()));
        cudaDeviceSynchronize();

        // Phase 2: sort by hash
        thrust::sort_by_key(d_hashes.begin(), d_hashes.end(), d_indices.begin());

        // Phase 3: find duplicate
        d_result[0] = (unsigned long long)~0ull;
        find_dup_kernel<<<find_blocks, tpb>>>(
            thrust::raw_pointer_cast(d_hashes.data()),
            thrust::raw_pointer_cast(d_indices.data()),
            batch_n,
            thrust::raw_pointer_cast(d_result.data())
        );
        cudaDeviceSynchronize();

        unsigned long long collision_count = d_result[0];
        *count_out += batch_n;

        if (collision_count != (unsigned long long)~0ull) {
            found = true;
        } else {
            base += batch_n;
        }
    }

    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(ms_out, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    static const char *algo_names[] = {"md5", "sha1", "sha256"};
    static const int   bits_arr[]   = {16, 24, 32, 40, 48};
    static const int   N_BITS       = 5;

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

    printf("# algo  bits  thrust_count  thrust_ms  expected\n");

    for (int a = a0; a < a1; a++) {
        for (int bi = b0; bi < b1; bi++) {
            int      bits     = bits_arr[bi];
            uint64_t expected = (uint64_t)(sqrt(M_PI / 2.0) * pow(2.0, bits / 2.0));

            float    ms    = 0.0f;
            uint64_t count = 0;
            thrust_collision(a, bits, &ms, &count);

            printf("%s %d %llu %.3f %llu\n",
                   algo_names[a], bits,
                   (unsigned long long)count, ms,
                   (unsigned long long)expected);
            fflush(stdout);
        }
    }
    return 0;
}
