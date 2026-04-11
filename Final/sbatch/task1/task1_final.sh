#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task1_final
#SBATCH -o Final/sbatch/task1/logs/task1_final_%j.out
#SBATCH -t 0-00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Single-run sanity check for Task 1 (serial + OpenMP + GPU)
# Usage: sbatch Final/sbatch/task1/task1_final.sh [n] [t] [tpb]
#   n    number of hashes per algorithm  (default: 10000000)
#   t    OpenMP thread count             (default: 4)
#   tpb  CUDA threads per block          (default: 256)
# Note: --cpus-per-task=4 to fit GPU nodes in the instruction partition.

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

N=${1:-10000000}
T=${2:-4}
TPB=${3:-256}

echo "=== Task 1 — serial + OpenMP + GPU (n=$N, t=$T, tpb=$TPB) ==="
echo "Host: $(hostname)  Cores: $SLURM_CPUS_PER_TASK"

module load nvidia/cuda/12.0 2>/dev/null || true

nvcc -O3 -std=c++17 -allow-unsupported-compiler -Xcompiler -fopenmp \
    -o Final/task1 \
    Final/Code/task1.cu \
    Final/Code/md5.cu \
    Final/Code/sha1.cu \
    Final/Code/sha256.cu \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp

echo "--- raw output ---"
./Final/task1 "$N" "$T" "$TPB"
