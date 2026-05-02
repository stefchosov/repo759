// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — Task 4: GPU collision search via Thrust parallel sort
//
// Three-phase algorithm:
//   Phase 1 (parallel kernel): hash batch_n inputs into device arrays
//   Phase 2 (Thrust):          thrust::sort_by_key on (hash, index) pairs
//   Phase 3 (parallel kernel): scan for first adjacent equal-hash pair
//
// Two memory modes:
//   --device  (default): cudaMalloc — pure VRAM. Fastest but capped by VRAM.
//   --unified:           cudaMallocManaged — pages migrate between GPU/host
//                        on demand. Lets us run beyond VRAM at the cost of
//                        PCIe-bound performance during the sort. Used to
//                        demonstrate the device-vs-unified inversion at large
//                        bit widths where data spills to host RAM.
//
// Adaptive over-allocation factor (controls retry rate):
//   bits <= 48: 4× expected — one-shot, ~99.9% find rate
//   bits = 52:  2× expected — ~63% find rate per round, avg 1.6 rounds
//   bits >= 56: 1× expected — ~50% find rate per round, avg 2× rounds
//
// Output columns: mode  algo  bits  thrust_count  thrust_ms  expected
//
// Usage: ./task4 [--device|--unified] [<algo>] [<bits>]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

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

// ── Allocator wrapper: cudaMalloc vs cudaMallocManaged ──────────────────────

template<typename T>
static T* alloc_buf(size_t n, bool unified) {
    T *p = nullptr;
    if (unified) cudaMallocManaged(&p, n * sizeof(T));
    else         cudaMalloc(&p, n * sizeof(T));
    return p;
}

// ── Adaptive over-allocation factor ──────────────────────────────────────────

static uint64_t batch_multiplier(int bits) {
    if (bits <= 48) return 4;
    if (bits <= 52) return 2;
    return 1;
}

// ── Thrust collision search ──────────────────────────────────────────────────

static void thrust_collision(int algo, int bits, bool unified,
                             float *ms_out, uint64_t *count_out) {
    const uint64_t hash_mask = (bits < 64) ? ((1ull << bits) - 1ull) : ~0ull;
    const uint64_t expected  = (uint64_t)(sqrt(M_PI / 2.0) * pow(2.0, bits / 2.0));
    const uint64_t mult      = batch_multiplier(bits);
    const uint64_t batch_n   = std::max((uint64_t)4096, expected * mult);

    *count_out = 0;
    bool     found = false;
    uint64_t base  = 0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    // Allocate once, reuse across rounds
    uint64_t           *d_hashes  = alloc_buf<uint64_t>(batch_n, unified);
    uint64_t           *d_indices = alloc_buf<uint64_t>(batch_n, unified);
    unsigned long long *d_result  = alloc_buf<unsigned long long>(1, unified);

    if (!d_hashes || !d_indices || !d_result) {
        fprintf(stderr, "ERROR: GPU allocation failed for batch_n=%llu (mode=%s, bits=%d)\n",
                (unsigned long long)batch_n, unified ? "unified" : "device", bits);
        if (d_hashes)  cudaFree(d_hashes);
        if (d_indices) cudaFree(d_indices);
        if (d_result)  cudaFree(d_result);
        *ms_out = -1.0f;
        return;
    }

    cudaEventRecord(ev0);

    const int tpb         = 256;
    const int hash_blocks = (int)((batch_n + tpb - 1) / tpb);
    const int find_blocks = (int)((batch_n + tpb - 1) / tpb);

    while (!found) {
        // Phase 1: hash
        if      (algo == 0) hash_to_arrays<0><<<hash_blocks, tpb>>>(
            hash_mask, base, batch_n, d_hashes, d_indices);
        else if (algo == 1) hash_to_arrays<1><<<hash_blocks, tpb>>>(
            hash_mask, base, batch_n, d_hashes, d_indices);
        else                hash_to_arrays<2><<<hash_blocks, tpb>>>(
            hash_mask, base, batch_n, d_hashes, d_indices);
        cudaDeviceSynchronize();

        // Phase 2: sort by hash (works on raw pointers via thrust::device_ptr;
        // identical for cudaMalloc and cudaMallocManaged buffers — the latter
        // simply pages in/out across PCIe as the radix sort scans the array).
        auto hkey = thrust::device_pointer_cast(d_hashes);
        auto hval = thrust::device_pointer_cast(d_indices);
        thrust::sort_by_key(hkey, hkey + batch_n, hval);

        // Phase 3: find duplicate
        unsigned long long sentinel = (unsigned long long)~0ull;
        cudaMemcpy(d_result, &sentinel, sizeof(sentinel), cudaMemcpyHostToDevice);
        find_dup_kernel<<<find_blocks, tpb>>>(
            d_hashes, d_indices, batch_n, d_result);
        cudaDeviceSynchronize();

        unsigned long long collision_count;
        cudaMemcpy(&collision_count, d_result, sizeof(collision_count),
                   cudaMemcpyDeviceToHost);
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

    cudaFree(d_hashes);
    cudaFree(d_indices);
    cudaFree(d_result);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    static const char *algo_names[] = {"md5", "sha1", "sha256"};
    // Wider sweep when running unified mode — the whole point is to push past VRAM
    static const int   bits_dev[]   = {16, 24, 32, 40, 48};
    static const int   N_DEV        = 5;
    static const int   bits_uni[]   = {16, 24, 32, 40, 48, 52, 56};
    static const int   N_UNI        = 7;

    bool unified = false;
    int  arg_i   = 1;
    if (argc > arg_i && strcmp(argv[arg_i], "--unified") == 0) { unified = true; arg_i++; }
    else if (argc > arg_i && strcmp(argv[arg_i], "--device") == 0) {            arg_i++; }

    const int *bits_arr = unified ? bits_uni : bits_dev;
    const int  N_BITS   = unified ? N_UNI    : N_DEV;

    int a0 = 0, a1 = 3, b0 = 0, b1 = N_BITS;
    if (argc > arg_i) {
        for (int a = 0; a < 3; a++)
            if (strcmp(argv[arg_i], algo_names[a]) == 0) { a0 = a; a1 = a + 1; }
        arg_i++;
    }
    if (argc > arg_i) {
        int bv = atoi(argv[arg_i]);
        for (int bi = 0; bi < N_BITS; bi++)
            if (bits_arr[bi] == bv) { b0 = bi; b1 = bi + 1; }
    }

    printf("# mode  algo  bits  thrust_count  thrust_ms  expected\n");
    fprintf(stderr, "Mode: %s\n", unified ? "unified" : "device");

    for (int a = a0; a < a1; a++) {
        for (int bi = b0; bi < b1; bi++) {
            int      bits     = bits_arr[bi];
            uint64_t expected = (uint64_t)(sqrt(M_PI / 2.0) * pow(2.0, bits / 2.0));

            float    ms    = 0.0f;
            uint64_t count = 0;
            thrust_collision(a, bits, unified, &ms, &count);

            printf("%s %s %d %llu %.3f %llu\n",
                   unified ? "unified" : "device",
                   algo_names[a], bits,
                   (unsigned long long)count, ms,
                   (unsigned long long)expected);
            fflush(stdout);
        }
    }
    return 0;
}
