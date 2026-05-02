#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task4_scaling
#SBATCH -o Final/sbatch/task4/logs/task4_scaling_%j.out
#SBATCH -t 0-00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

# Task 4 — GPU collision search via Thrust parallel sort
# Runs two modes back-to-back:
#   --device  : pure VRAM, capped at bits=48 (working set fits in 8 GB GPU)
#   --unified : cudaMallocManaged, pages spill to host RAM on demand,
#               extends sweep to bits in {52, 56} to demonstrate the
#               PCIe-bound performance inversion vs Pollard's rho.
# Need --mem=96G on the host because the unified-memory sort spills there.
#
# Produces:
#   Final/Data/task4/scaling_task4_device.dat
#   Final/Data/task4/scaling_task4_unified.dat

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

mkdir -p Final/Data/task4
DEV=Final/Data/task4/scaling_task4_device.dat
UNI=Final/Data/task4/scaling_task4_unified.dat

echo "=== Task 4 Thrust GPU collision scaling — $(date) ==="
echo "Host: $(hostname)"

module load nvidia/cuda/13.0.0
echo "CUDA: $(nvcc --version | grep release)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

nvcc -O3 -std=c++17 \
    -o Final/task4 \
    Final/Code/task4.cu

echo "=== Running --device (bits 16..48, 4× allocation) ==="
./Final/task4 --device | tee "$DEV"

echo "=== Running --unified (bits 16..56, adaptive 4×/2×/1× allocation) ==="
./Final/task4 --unified | tee "$UNI"

echo "=== Done ==="
