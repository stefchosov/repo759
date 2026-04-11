// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Final project — CPU MD5 implementation (RFC 1321)

#pragma once
#include <cstddef>
#include <cstdint>

// Compute MD5 of 'len' bytes at 'msg'.  Result written to digest[0..15].
void md5_cpu(const uint8_t *msg, size_t len, uint8_t digest[16]);
