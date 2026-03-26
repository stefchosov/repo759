// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW09.pdf

#include "cluster.h"
#include <cmath>

// False sharing fix: accumulate into a thread-local variable so that the
// shared dists[] array is only written once per thread (at the end), rather
// than on every loop iteration.  The original code wrote dists[tid] in a
// tight inner loop, causing cache-line ping-ponging between threads.
void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
#pragma omp parallel num_threads(t)
  {
    unsigned int tid = omp_get_thread_num();
    float local_dist = 0.0f;
#pragma omp for
    for (size_t i = 0; i < n; i++) {
      local_dist += std::fabs(arr[i] - centers[tid]);
    }
    dists[tid] = local_dist;
  }
}
