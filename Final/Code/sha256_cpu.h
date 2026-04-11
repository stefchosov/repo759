// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — CPU SHA-256 implementation (FIPS 180-4)

#pragma once
#include <cstddef>
#include <cstdint>

// Compute SHA-256 of 'len' bytes at 'msg'.  Result written to digest[0..31].
void sha256_cpu(const uint8_t *msg, size_t len, uint8_t digest[32]);
