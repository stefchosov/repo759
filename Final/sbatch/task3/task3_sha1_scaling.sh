#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task3_sha1
#SBATCH -o Final/sbatch/task3/logs/task3_sha1_scaling_%j.out
#SBATCH -t 0-01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

# Task 3 scaling — SHA-1, bits in {16,24,32,40,48,56,64}
# CPU serial + OpenMP 12-thread (single-socket sweet spot) for bits <= 56; GPU for all bits.
# 12 threads chosen: OMP scaling test showed peak at 12 cores (NUMA boundary on dual-socket node).
# Produces: Final/Data/task3/scaling_task3_sha1.dat

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

mkdir -p Final/Data/task3
OUTFILE=Final/Data/task3/scaling_task3_sha1.dat

echo "=== Task 3 SHA-1 scaling — $(date) ==="
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

echo "=== Running SHA-1, bits = {16 24 32 40 48 56 64} ==="
OMP_NUM_THREADS=12 ./Final/task3 sha1 | tee "$OUTFILE"

echo "=== Written to $OUTFILE ==="
