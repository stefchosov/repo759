#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task2_final
#SBATCH -o Final/sbatch/task2/logs/task2_final_%j.out
#SBATCH -t 0-00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Single-run sanity check for Task 2 (GPU tpb scaling)
# Usage: sbatch Final/sbatch/task2/task2_final.sh [n] [tpb]
#   n    number of hashes per algorithm  (default: 10000000)
#   tpb  threads per block               (default: 256)

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

N=${1:-10000000}
TPB=${2:-256}

echo "=== Task 2 — GPU tpb scaling (n=$N, tpb=$TPB) ==="
echo "Host: $(hostname)"

module load nvidia/cuda/13.0.0
echo "CUDA: $(nvcc --version | grep release)"

nvcc -O3 -std=c++17 \
    -o Final/task2 \
    Final/Code/task2.cu \
    Final/Code/md5.cu \
    Final/Code/sha1.cu \
    Final/Code/sha256.cu \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp

echo "--- raw output ---"
./Final/task2 "$N" "$TPB"
