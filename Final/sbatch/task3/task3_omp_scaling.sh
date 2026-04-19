#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task3_omp_scaling
#SBATCH -o Final/sbatch/task3/logs/task3_omp_scaling_%j.out
#SBATCH -t 0-00:20:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# OpenMP thread-count scaling: sweep OMP_NUM_THREADS in {4,8,12,16,20,24,28,32}
# Fixed algo=md5, bits in {40, 48} — fast enough for multiple runs, big enough to show scaling.
# Produces: Final/Data/task3/omp_scaling.dat
#   columns: threads  bits  omp_ms

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

mkdir -p Final/Data/task3
OUTFILE=Final/Data/task3/omp_scaling.dat

echo "=== Task 3 OpenMP thread scaling — $(date) ==="
echo "Host: $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "nproc: $(nproc)"

module load nvidia/cuda/13.0.0
echo "CUDA: $(nvcc --version | grep release)"

nvcc -O3 -std=c++17 \
    -Xcompiler -fopenmp \
    -o Final/task3 \
    Final/Code/task3.cu \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp

echo "# threads  bits  omp_ms" | tee "$OUTFILE"

for THREADS in 1 2 4 8 12 16 20 24 28 32; do
    for BITS in 40 48; do
        echo -n "threads=$THREADS bits=$BITS ... " >&2
        # Extract omp_ms column (field 6) from the data line
        OMP_NUM_THREADS=$THREADS ./Final/task3 md5 $BITS 2>/dev/null \
            | grep -v '^#' \
            | awk -v t=$THREADS '{printf "%d %d %s\n", t, $2, $6}' \
            | tee -a "$OUTFILE"
    done
done

echo "=== Written to $OUTFILE ==="
