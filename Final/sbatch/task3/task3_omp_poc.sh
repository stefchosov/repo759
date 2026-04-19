#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task3_omp_poc
#SBATCH -o Final/sbatch/task3/logs/task3_omp_poc_%j.out
#SBATCH -t 0-00:05:00
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# POC: discover how many CPUs Euler actually allocates and what OpenMP
# thread counts are meaningful. Prints nproc, OMP_NUM_THREADS behavior,
# and runs a trivial parallel loop to confirm thread counts.

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

echo "=== OpenMP POC — $(date) ==="
echo "Host: $(hostname)"
echo "Requested cpus-per-task: 40"
echo "Actual nproc: $(nproc)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

module load nvidia/cuda/13.0.0

# Compile a minimal OpenMP test
cat > /tmp/omp_test.cpp << 'EOF'
#include <cstdio>
#include <omp.h>
int main() {
    int ts[] = {1, 2, 4, 8, 16, 20, 32, 40};
    for (int t : ts) {
        omp_set_num_threads(t);
        int seen = 0;
        #pragma omp parallel reduction(+:seen)
        { seen += 1; }
        printf("requested=%2d  actual=%2d\n", t, seen);
    }
}
EOF
g++ -O2 -fopenmp -o /tmp/omp_test /tmp/omp_test.cpp
echo "--- thread availability ---"
/tmp/omp_test
