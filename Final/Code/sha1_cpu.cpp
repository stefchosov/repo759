// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — CPU SHA-1 implementation (FIPS 180-4)

#include "sha1_cpu.h"
#include <cstring>

static inline uint32_t rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32u - n));
}

// Process one 64-byte block; state is updated in place.
static void sha1_block(const uint8_t blk[64], uint32_t st[5]) {
    uint32_t W[80];
    for (int i = 0; i < 16; i++)
        W[i] = ((uint32_t)blk[4*i]   << 24)
             | ((uint32_t)blk[4*i+1] << 16)
             | ((uint32_t)blk[4*i+2] <<  8)
             | ((uint32_t)blk[4*i+3]);
    for (int i = 16; i < 80; i++)
        W[i] = rotl32(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1);

    uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4];

    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & c) | (~b & d);
            k = 0x5a827999u;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ c ^ d;
            k = 0xca62c1d6u;
        }
        uint32_t tmp = rotl32(a, 5) + f + e + k + W[i];
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = tmp;
    }

    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e;
}

void sha1_cpu(const uint8_t *msg, size_t len, uint8_t digest[20]) {
    uint32_t st[5] = {
        0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u
    };

    size_t i = 0;
    for (; i + 64 <= len; i += 64)
        sha1_block(msg + i, st);

    uint8_t blk[64] = {};
    size_t rem = len - i;
    memcpy(blk, msg + i, rem);
    blk[rem] = 0x80;

    if (rem >= 56) {
        sha1_block(blk, st);
        memset(blk, 0, 64);
    }

    // Append original length in bits as 64-bit big-endian
    uint64_t bits = (uint64_t)len * 8;
    for (int j = 0; j < 8; j++)
        blk[56 + j] = (uint8_t)(bits >> (56 - 8*j));
    sha1_block(blk, st);

    // Output: five 32-bit words in big-endian order
    for (int j = 0; j < 5; j++) {
        digest[4*j]   = (uint8_t)(st[j] >> 24);
        digest[4*j+1] = (uint8_t)(st[j] >> 16);
        digest[4*j+2] = (uint8_t)(st[j] >>  8);
        digest[4*j+3] = (uint8_t)(st[j]);
    }
}
