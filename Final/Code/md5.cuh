// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — GPU MD5 kernel interface

#pragma once
#include <cstdint>

// Compute n MD5 hashes on the GPU, one per thread.
// Input for thread i: uint64_t i encoded as 8-byte little-endian.
// digests: device memory, size n*16 bytes.
// tpb: threads per block.
// Calls cudaDeviceSynchronize() before returning.
void md5_gpu(uint64_t n, uint8_t *digests, int tpb);
