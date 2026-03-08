// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW07.pdf

#include "count.cuh"
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

// Finds unique integers in d_in and their occurrence counts.
// values is filled with unique integers in ascending order.
// counts is filled with the corresponding occurrence counts.
//
// Algorithm:
//   1. Copy and sort d_in.
//   2. Use thrust::reduce_by_key over a ones-array to count runs of equal keys.
//   3. Resize and populate the output vectors.
void count(const thrust::device_vector<int> &d_in,
           thrust::device_vector<int> &values,
           thrust::device_vector<int> &counts) {
    const int n = static_cast<int>(d_in.size());

    // 1. Sorted copy of the input
    thrust::device_vector<int> sorted(d_in);
    thrust::sort(sorted.begin(), sorted.end());

    // 2. All-ones array for counting
    thrust::device_vector<int> ones(n, 1);

    // 3. reduce_by_key collapses adjacent equal keys, summing the ones
    thrust::device_vector<int> tmp_vals(n);
    thrust::device_vector<int> tmp_counts(n);
    auto ends = thrust::reduce_by_key(sorted.begin(), sorted.end(),
                                      ones.begin(),
                                      tmp_vals.begin(),
                                      tmp_counts.begin());

    const int num_unique = static_cast<int>(ends.first - tmp_vals.begin());

    // 4. Write results into caller-provided output vectors
    values.resize(num_unique);
    counts.resize(num_unique);
    thrust::copy(tmp_vals.begin(),   tmp_vals.begin()   + num_unique, values.begin());
    thrust::copy(tmp_counts.begin(), tmp_counts.begin() + num_unique, counts.begin());
}
