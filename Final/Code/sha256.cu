// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — GPU SHA-256 kernel (FIPS 180-4)
//
// Each thread computes SHA-256 of a fixed 8-byte input: the thread's uint64_t
// index encoded as little-endian bytes.  The padded block is built directly
// as big-endian uint32 words to match SHA-256's byte order.

#include "sha256.cuh"
#include <cuda_runtime.h>

__constant__ static uint32_t c_sha256_K[64] = {
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

__device__ static inline uint32_t rotr32d(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

__device__ static inline uint32_t bswap32d(uint32_t x) {
    return ((x & 0xffu) << 24) | ((x & 0xff00u) << 8)
         | ((x >> 8)  & 0xff00u) | ((x >> 24) & 0xffu);
}

__device__ static void sha256_compress(const uint32_t W0[16], uint32_t st[8]) {
    uint32_t W[64];
    for (int i = 0;  i < 16; i++) W[i] = W0[i];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32d(W[i-15],  7) ^ rotr32d(W[i-15], 18) ^ (W[i-15] >>  3);
        uint32_t s1 = rotr32d(W[i- 2], 17) ^ rotr32d(W[i- 2], 19) ^ (W[i- 2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    uint32_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint32_t e = st[4], f = st[5], g = st[6], h = st[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1    = rotr32d(e,  6) ^ rotr32d(e, 11) ^ rotr32d(e, 25);
        uint32_t ch    = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + c_sha256_K[i] + W[i];
        uint32_t S0    = rotr32d(a,  2) ^ rotr32d(a, 13) ^ rotr32d(a, 22);
        uint32_t maj   = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d;
    st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

// One thread computes SHA-256(LE-uint64(i)) and writes 32 bytes to digests[i*32].
__global__ void sha256_kernel(uint64_t n, uint8_t *digests) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Same word layout as SHA-1 (both big-endian):
    // W[0]=bswap32(lower32(i)), W[1]=bswap32(upper32(i))
    // W[2]=0x80000000, W[3..14]=0, W[15]=0x00000040
    uint32_t W0[16] = {};
    W0[0]  = bswap32d((uint32_t)(i));
    W0[1]  = bswap32d((uint32_t)(i >> 32));
    W0[2]  = 0x80000000u;
    W0[15] = 0x00000040u;

    uint32_t st[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    sha256_compress(W0, st);

    uint8_t *d = digests + i * 32u;
    for (int j = 0; j < 8; j++) {
        d[4*j]   = (uint8_t)(st[j] >> 24);
        d[4*j+1] = (uint8_t)(st[j] >> 16);
        d[4*j+2] = (uint8_t)(st[j] >>  8);
        d[4*j+3] = (uint8_t)(st[j]);
    }
}

void sha256_gpu(uint64_t n, uint8_t *digests, int tpb) {
    int blocks = (int)((n + (uint64_t)tpb - 1) / (uint64_t)tpb);
    sha256_kernel<<<blocks, tpb>>>(n, digests);
    cudaDeviceSynchronize();
}
