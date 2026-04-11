// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — GPU SHA-1 kernel (FIPS 180-4)
//
// Each thread computes SHA-1 of a fixed 8-byte input: the thread's uint64_t
// index encoded as little-endian bytes.  The padded block is built directly
// as big-endian uint32 words to match SHA-1's byte order.

#include "sha1.cuh"
#include <cuda_runtime.h>

__device__ static inline uint32_t rotl32d(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32u - n));
}

// Byte-swap a 32-bit word (LE → BE or vice versa).
__device__ static inline uint32_t bswap32d(uint32_t x) {
    return ((x & 0xffu) << 24) | ((x & 0xff00u) << 8)
         | ((x >> 8)  & 0xff00u) | ((x >> 24) & 0xffu);
}

// Compress one block; W0 holds the 16 initial big-endian message words.
__device__ static void sha1_compress(const uint32_t W0[16], uint32_t st[5]) {
    uint32_t W[80];
    for (int i = 0;  i < 16; i++) W[i] = W0[i];
    for (int i = 16; i < 80; i++)
        W[i] = rotl32d(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1);

    uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if      (i < 20) { f = (b & c) | (~b & d);          k = 0x5a827999u; }
        else if (i < 40) { f = b ^ c ^ d;                   k = 0x6ed9eba1u; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdcu; }
        else             { f = b ^ c ^ d;                    k = 0xca62c1d6u; }
        uint32_t tmp = rotl32d(a, 5) + f + e + k + W[i];
        e = d; d = c; c = rotl32d(b, 30); b = a; a = tmp;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e;
}

// One thread computes SHA-1(LE-uint64(i)) and writes 20 bytes to digests[i*20].
__global__ void sha1_kernel(uint64_t n, uint8_t *digests) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Build padded message schedule as big-endian uint32 words.
    // 8-byte LE input reinterpreted as two BE words:
    //   W[0] = bswap32(lower32(i))   bytes 0..3 stored BE
    //   W[1] = bswap32(upper32(i))   bytes 4..7 stored BE
    // Padding 0x80 at byte 8 → W[2] = 0x80000000
    // Zero fill W[3..13]
    // 64-bit BE bit-length (=64=0x40) at bytes 56..63 → W[14]=0, W[15]=0x00000040
    uint32_t W0[16] = {};
    W0[0]  = bswap32d((uint32_t)(i));
    W0[1]  = bswap32d((uint32_t)(i >> 32));
    W0[2]  = 0x80000000u;
    W0[15] = 0x00000040u;

    uint32_t st[5] = {
        0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u
    };
    sha1_compress(W0, st);

    uint8_t *d = digests + i * 20u;
    for (int j = 0; j < 5; j++) {
        d[4*j]   = (uint8_t)(st[j] >> 24);
        d[4*j+1] = (uint8_t)(st[j] >> 16);
        d[4*j+2] = (uint8_t)(st[j] >>  8);
        d[4*j+3] = (uint8_t)(st[j]);
    }
}

void sha1_gpu(uint64_t n, uint8_t *digests, int tpb) {
    int blocks = (int)((n + (uint64_t)tpb - 1) / (uint64_t)tpb);
    sha1_kernel<<<blocks, tpb>>>(n, digests);
    cudaDeviceSynchronize();
}
