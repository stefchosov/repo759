// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — GPU MD5 kernel (RFC 1321)
//
// Each thread computes MD5 of a fixed 8-byte input: the thread's uint64_t
// index encoded as little-endian bytes.  The padded block is constructed
// directly as 16 uint32 words, avoiding any byte-level manipulation.

#include "md5.cuh"
#include <cuda_runtime.h>

// Per-round constants in constant memory (all threads read same value per round)
__constant__ static uint32_t c_md5_K[64] = {
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

__constant__ static uint32_t c_md5_S[64] = {
     7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
     5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
     4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
     6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21
};

__device__ static inline uint32_t rotl32d(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32u - n));
}

// Compress one block given as 16 pre-decoded little-endian words.
__device__ static void md5_compress(const uint32_t M[16], uint32_t st[4]) {
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3];
    for (int i = 0; i < 64; i++) {
        uint32_t f, g;
        if      (i < 16) { f = (b & c) | (~b & d); g = (uint32_t)i;         }
        else if (i < 32) { f = (d & b) | (~d & c); g = (5u*i + 1u) % 16u;   }
        else if (i < 48) { f = b ^ c ^ d;           g = (3u*i + 5u) % 16u;   }
        else             { f = c ^ (b | ~d);          g = (7u*i)      % 16u;   }
        f += a + c_md5_K[i] + M[g];
        a = d; d = c; c = b; b += rotl32d(f, c_md5_S[i]);
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
}

// One thread computes MD5(LE-uint64(i)) and writes 16 bytes to digests[i*16].
__global__ void md5_kernel(uint64_t n, uint8_t *digests) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Build padded message schedule directly as LE uint32 words.
    // 8-byte input: M[0]=lower32(i), M[1]=upper32(i)
    // Padding byte 0x80 at position 8 → M[2] = 0x00000080
    // Zero fill through M[13]
    // 64-bit LE bit-length (= 64 = 0x40) at bytes 56–63 → M[14] = 0x00000040, M[15] = 0
    uint32_t M[16] = {};
    M[0]  = (uint32_t)(i);
    M[1]  = (uint32_t)(i >> 32);
    M[2]  = 0x00000080u;
    M[14] = 0x00000040u;

    uint32_t st[4] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};
    md5_compress(M, st);

    uint8_t *d = digests + i * 16u;
    for (int j = 0; j < 4; j++) {
        d[4*j]   = (uint8_t)(st[j]);
        d[4*j+1] = (uint8_t)(st[j] >>  8);
        d[4*j+2] = (uint8_t)(st[j] >> 16);
        d[4*j+3] = (uint8_t)(st[j] >> 24);
    }
}

void md5_gpu(uint64_t n, uint8_t *digests, int tpb) {
    int blocks = (int)((n + (uint64_t)tpb - 1) / (uint64_t)tpb);
    md5_kernel<<<blocks, tpb>>>(n, digests);
    cudaDeviceSynchronize();
}
