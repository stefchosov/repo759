// Code generated with assistance from Claude Code (Anthropic CLI)
// Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
// Usage: Implementation based on problem specification from HW08.pdf

#include "msort.h"
#include <algorithm>

// Merges arr[lo..mid) and arr[mid..hi) using tmp as scratch, writing back to arr.
static void merge_arrays(int *arr, int *tmp,
                         std::size_t lo, std::size_t mid, std::size_t hi) {
    std::size_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi)
        tmp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i < mid) tmp[k++] = arr[i++];
    while (j < hi)  tmp[k++] = arr[j++];
    std::copy(tmp + lo, tmp + hi, arr + lo);
}

// Recursive parallel merge sort using OpenMP tasks.
// arr[lo..hi) is sorted in place.
// Below threshold, fall through to std::sort (serial, no recursion overhead).
static void msort_rec(int *arr, int *tmp,
                      std::size_t lo, std::size_t hi,
                      std::size_t threshold) {
    if (hi - lo <= threshold) {
        std::sort(arr + lo, arr + hi);
        return;
    }

    std::size_t mid = lo + (hi - lo) / 2;

    #pragma omp task
    msort_rec(arr, tmp, lo, mid, threshold);

    #pragma omp task
    msort_rec(arr, tmp, mid, hi, threshold);

    #pragma omp taskwait

    merge_arrays(arr, tmp, lo, mid, hi);
}

// Public entry point. Creates the parallel region and launches the sort.
// Thread count is controlled by the caller via omp_set_num_threads().
void msort(int *arr, const std::size_t n, const std::size_t threshold) {
    int *tmp = new int[n];

    #pragma omp parallel
    {
        #pragma omp single nowait
        msort_rec(arr, tmp, 0, n, threshold);
    }

    delete[] tmp;
}
