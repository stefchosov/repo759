#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task2_final_scaling
#SBATCH -o Final/sbatch/task2/logs/task2_final_scaling_%j.out
#SBATCH -t 0-00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Scaling study for Task 2 — sweeps tpb at fixed n=10M.
# Produces: Final/Data/task2/scaling_task2.dat
#
# Columns:
#   tpb  md5_gpu_ms  sha1_gpu_ms  sha256_gpu_ms

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

N=10000000
OUTFILE=Final/Data/task2/scaling_task2.dat

echo "=== Task 2 scaling (n=$N) — $(date) ==="
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

echo "# tpb  md5_gpu_ms  sha1_gpu_ms  sha256_gpu_ms" > "$OUTFILE"

for TPB in 32 64 128 256 512 1024; do
    echo -n "  tpb=$TPB ... "
    RAW=$(./Final/task2 "$N" "$TPB")

    MD5_MS=$(echo  "$RAW" | sed -n '2p')
    SHA1_MS=$(echo "$RAW" | sed -n '4p')
    S256_MS=$(echo "$RAW" | sed -n '6p')

    echo "$TPB  $MD5_MS  $SHA1_MS  $S256_MS" >> "$OUTFILE"
    echo "done"
done

echo "=== Written to $OUTFILE ==="
cat "$OUTFILE"
