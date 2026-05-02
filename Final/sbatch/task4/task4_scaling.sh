#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task4_scaling
#SBATCH -o Final/sbatch/task4/logs/task4_scaling_%j.out
#SBATCH -t 0-00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Task 4 — GPU collision search via Thrust parallel sort
# Sweeps {md5, sha1, sha256} × bits in {16,24,32,40,48}.
# Capped at bits=48 because the sort working set (4 * expected * 16 bytes)
# saturates 8 GB GPU VRAM at bits >= 52.
# Produces: Final/Data/task4/scaling_task4.dat

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

mkdir -p Final/Data/task4
OUTFILE=Final/Data/task4/scaling_task4.dat

echo "=== Task 4 Thrust GPU collision scaling — $(date) ==="
echo "Host: $(hostname)"

module load nvidia/cuda/13.0.0
echo "CUDA: $(nvcc --version | grep release)"

nvcc -O3 -std=c++17 \
    -o Final/task4 \
    Final/Code/task4.cu

echo "=== Running task4: 3 algos × 5 bits ==="
./Final/task4 | tee "$OUTFILE"

echo "=== Written to $OUTFILE ==="
