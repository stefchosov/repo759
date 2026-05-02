// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
//
// Shared device-side narrow (uint64_t) hash functions for Task 3 / Task 4.
// All declarations are file-static so multiple .cu files can include this
// header without ODR conflicts; each translation unit gets its own copy in
// __constant__ memory.
//
// Truncated extraction (matching CPU side):
//   MD5 (LE output):  bytes 0-7 = state[0] | (state[1] << 32)
//   SHA (BE output):  bytes 0-7 = bswap32(state[0]) | (bswap32(state[1]) << 32)

#pragma once
#include <cstdint>

// ── GPU constant memory ──────────────────────────────────────────────────────

__constant__ static uint32_t gh_md5_K[64] = {
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
__constant__ static uint32_t gh_md5_S[64] = {
     7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
     5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
     4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
     6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21
};
__constant__ static uint32_t gh_sha256_K[64] = {
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

__device__ static inline uint32_t gh_rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32u - n));
}
__device__ static inline uint32_t gh_rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}
__device__ static inline uint32_t gh_bswap32(uint32_t x) {
    return ((x & 0xffu) << 24) | ((x & 0xff00u) << 8)
         | ((x >> 8)  & 0xff00u) | ((x >> 24) & 0xffu);
}

// ── Per-algorithm device hash functions: 8-byte input, 8-byte truncated LE output ──

__device__ static uint64_t gh_md5_le64(uint64_t i) {
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
        f += a + gh_md5_K[j] + M[g];
        a = d; d = c; c = b; b += gh_rotl32(f, gh_md5_S[j]);
    }
    return (uint64_t)(a0 + a) | ((uint64_t)(b0 + b) << 32);
}

__device__ static uint64_t gh_sha1_le64(uint64_t i) {
    uint32_t W[80] = {};
    W[0]  = gh_bswap32((uint32_t)(i));
    W[1]  = gh_bswap32((uint32_t)(i >> 32));
    W[2]  = 0x80000000u;
    W[15] = 0x00000040u;
    for (int j = 16; j < 80; j++)
        W[j] = gh_rotl32(W[j-3] ^ W[j-8] ^ W[j-14] ^ W[j-16], 1);

    uint32_t h0=0x67452301u, h1=0xefcdab89u;
    uint32_t a=h0, b=h1, c=0x98badcfeu, d=0x10325476u, e=0xc3d2e1f0u;
    for (int j = 0; j < 80; j++) {
        uint32_t f, k;
        if      (j < 20) { f = (b & c) | (~b & d);          k = 0x5a827999u; }
        else if (j < 40) { f = b ^ c ^ d;                   k = 0x6ed9eba1u; }
        else if (j < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdcu; }
        else             { f = b ^ c ^ d;                    k = 0xca62c1d6u; }
        uint32_t tmp = gh_rotl32(a, 5) + f + e + k + W[j];
        e = d; d = c; c = gh_rotl32(b, 30); b = a; a = tmp;
    }
    return (uint64_t)gh_bswap32(h0 + a) | ((uint64_t)gh_bswap32(h1 + b) << 32);
}

__device__ static uint64_t gh_sha256_le64(uint64_t i) {
    uint32_t W[64] = {};
    W[0]  = gh_bswap32((uint32_t)(i));
    W[1]  = gh_bswap32((uint32_t)(i >> 32));
    W[2]  = 0x80000000u;
    W[15] = 0x00000040u;
    for (int j = 16; j < 64; j++) {
        uint32_t s0 = gh_rotr32(W[j-15],  7) ^ gh_rotr32(W[j-15], 18) ^ (W[j-15] >>  3);
        uint32_t s1 = gh_rotr32(W[j- 2], 17) ^ gh_rotr32(W[j- 2], 19) ^ (W[j- 2] >> 10);
        W[j] = W[j-16] + s0 + W[j-7] + s1;
    }
    uint32_t h0=0x6a09e667u, h1=0xbb67ae85u;
    uint32_t a=h0,  b=h1,  c=0x3c6ef372u, d=0xa54ff53au;
    uint32_t e=0x510e527fu, f=0x9b05688cu, g=0x1f83d9abu, h=0x5be0cd19u;
    for (int j = 0; j < 64; j++) {
        uint32_t S1  = gh_rotr32(e,  6) ^ gh_rotr32(e, 11) ^ gh_rotr32(e, 25);
        uint32_t ch  = (e & f) ^ (~e & g);
        uint32_t t1  = h + S1 + ch + gh_sha256_K[j] + W[j];
        uint32_t S0  = gh_rotr32(a,  2) ^ gh_rotr32(a, 13) ^ gh_rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2  = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    return (uint64_t)gh_bswap32(h0 + a) | ((uint64_t)gh_bswap32(h1 + b) << 32);
}
