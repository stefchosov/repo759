#include "montecarlo.h"

// Returns the number of (x[i], y[i]) points that lie inside the circle of
// the given radius.  Uses OpenMP parallel for with the simd directive so the
// inner loop can be vectorised.  Compile with -DNOSIMD to omit the simd
// clause (used by the sbatch scaling script to generate a without-simd curve).
int montecarlo(const size_t n, const float *x, const float *y,
               const float radius) {
    int count = 0;
    const float r2 = radius * radius;

#ifdef NOSIMD
    #pragma omp parallel for reduction(+:count)
#else
    #pragma omp parallel for simd reduction(+:count)
#endif
    for (size_t i = 0; i < n; i++) {
        if (x[i] * x[i] + y[i] * y[i] <= r2) count++;
    }
    return count;
}
