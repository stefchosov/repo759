#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task3_md5
#SBATCH -o Final/sbatch/task3/logs/task3_md5_scaling_%j.out
#SBATCH -t 0-00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Task 3 scaling — MD5, bits in {16,24,32,40,48,56,64}
# CPU run for bits <= 56; GPU (Pollard's rho) for all bits.
# Produces: Final/Data/task3/scaling_task3_md5.dat

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

OUTFILE=Final/Data/task3/scaling_task3_md5.dat

echo "=== Task 3 MD5 scaling — $(date) ==="
echo "Host: $(hostname)"

module load nvidia/cuda/13.0.0
echo "CUDA: $(nvcc --version | grep release)"

nvcc -O3 -std=c++17 \
    -o Final/task3 \
    Final/Code/task3.cu \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp

echo "=== Running MD5, bits = {16 24 32 40 48 56 64} ==="
./Final/task3 md5 | tee "$OUTFILE"

echo "=== Written to $OUTFILE ==="
