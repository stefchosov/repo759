#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task3_final_scaling
#SBATCH -o Final/sbatch/task3/logs/task3_final_scaling_%j.out
#SBATCH -t 0-00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Full Task 3 experiment — all (algo, bits) combinations.
# Produces: Final/Data/task3/scaling_task3.dat
#
# Columns:
#   algo  bits  cpu_count  cpu_ms  gpu_batch  gpu_ms  expected

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

OUTFILE=Final/Data/task3/scaling_task3.dat

echo "=== Task 3 collision scaling — $(date) ==="
echo "Host: $(hostname)"

module load nvidia/cuda/13.0.0
echo "CUDA: $(nvcc --version | grep release)"

nvcc -O3 -std=c++17 \
    -o Final/task3 \
    Final/Code/task3.cu \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp

echo "=== Running all (algo, bits) combinations ==="
./Final/task3 | tee "$OUTFILE"

echo "=== Written to $OUTFILE ==="
