#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task3_final
#SBATCH -o Final/sbatch/task3/logs/task3_final_%j.out
#SBATCH -t 0-00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Sanity check for Task 3 — runs MD5 at 16 bits (fastest combination).
# Usage: sbatch Final/sbatch/task3/task3_final.sh [algo] [bits]
#   algo  algorithm name  (default: md5)
#   bits  truncated bits  (default: 16)

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

ALGO=${1:-md5}
BITS=${2:-16}

echo "=== Task 3 sanity check ($ALGO, ${BITS}-bit) — $(date) ==="
echo "Host: $(hostname)"

module load nvidia/cuda/13.0.0
echo "CUDA: $(nvcc --version | grep release)"

nvcc -O3 -std=c++17 \
    -o Final/task3 \
    Final/Code/task3.cu \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp

echo "--- output ---"
./Final/task3 "$ALGO" "$BITS"
