#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task4_scaling
#SBATCH -o Final/sbatch/task4/logs/task4_scaling_%j.out
#SBATCH -t 0-00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

# Task 4 — GPU collision search via Thrust parallel sort
#
# Two memory modes, both with 10 trials for consistent methodology:
#   --device  : pure VRAM, capped at bits=48. 10 trials — device mode runs
#               are cheap (<1 sec total) so trial repetition is essentially
#               free, and the residual variance from retry geometry (~1.8%
#               miss rate at 4× allocation) plus GPU clock/scheduler jitter
#               is worth characterizing.
#   --unified : cudaMallocManaged. 10 trials — geometric retry distribution
#               from sub-1× over-allocation at bits >= 52 makes single-trial
#               results unrepresentative.
#
# Adaptive over-allocation: 4×/2×/1×/0.5× for bits in {≤48, 52, 56, 64}.
# Need --mem=96G on the host for the bits>=56 unified-memory spill area.
#
# Produces:
#   Final/Data/task4/scaling_task4_device.dat  (1 trial per row)
#   Final/Data/task4/scaling_task4_unified.dat (10 trials per row, prefixed
#                                                with trial index)

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

mkdir -p Final/Data/task4
DEV=Final/Data/task4/scaling_task4_device.dat
UNI=Final/Data/task4/scaling_task4_unified.dat
N_TRIALS=10

echo "=== Task 4 Thrust GPU collision scaling — $(date) ==="
echo "Host: $(hostname)"

module load nvidia/cuda/13.0.0
echo "CUDA: $(nvcc --version | grep release)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

nvcc -O3 -std=c++17 \
    -o Final/task4 \
    Final/Code/task4.cu

echo "=== Running --device (${N_TRIALS} trials, bits 16..48) ==="
echo "# trial mode algo bits count ms expected" > "$DEV"
for trial in $(seq 1 $N_TRIALS); do
    echo "  device trial $trial / $N_TRIALS"
    ./Final/task4 --device 2>/dev/null \
        | awk -v t=$trial '/^[a-z]/ {print t, $0}' >> "$DEV"
done

echo "=== Running --unified (${N_TRIALS} trials, bits 16..64) ==="
echo "# trial mode algo bits count ms expected" > "$UNI"
for trial in $(seq 1 $N_TRIALS); do
    echo "  unified trial $trial / $N_TRIALS"
    ./Final/task4 --unified 2>/dev/null \
        | awk -v t=$trial '/^[a-z]/ {print t, $0}' >> "$UNI"
done

echo "=== Done ==="
echo "Device data: $(wc -l < $DEV) lines"
echo "Unified data: $(wc -l < $UNI) lines"
