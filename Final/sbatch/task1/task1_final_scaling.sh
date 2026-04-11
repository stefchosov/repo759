#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task1_final_scaling
#SBATCH -o Final/sbatch/task1/logs/task1_final_scaling_%j.out
#SBATCH -t 0-00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Scaling study for Task 1 — varies n, fixed t=4, tpb=256.
# Produces: Final/Data/task1/scaling_task1.dat
#
# Columns:
#   n  md5_serial_ms  md5_omp_ms  md5_gpu_ms
#   sha1_serial_ms  sha1_omp_ms  sha1_gpu_ms
#   sha256_serial_ms  sha256_omp_ms  sha256_gpu_ms
# Note: --cpus-per-task=4 to fit GPU nodes in the instruction partition.

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

T=4
TPB=256
OUTFILE=Final/Data/task1/scaling_task1.dat

echo "=== Task 1 scaling (t=$T, tpb=$TPB) — $(date) ==="
echo "Host: $(hostname)"

module load nvidia/cuda/12.5 2>/dev/null || \
module load nvidia/cuda/12.3 2>/dev/null || \
module load nvidia/cuda/12.2 2>/dev/null || \
module load nvidia/cuda/12.0 2>/dev/null || true
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"

nvcc -O3 -std=c++17 -Xcompiler -fopenmp \
    -o Final/task1 \
    Final/Code/task1.cu \
    Final/Code/md5.cu \
    Final/Code/sha1.cu \
    Final/Code/sha256.cu \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp

echo "# n  md5_serial_ms  md5_omp_ms  md5_gpu_ms  sha1_serial_ms  sha1_omp_ms  sha1_gpu_ms  sha256_serial_ms  sha256_omp_ms  sha256_gpu_ms" \
    > "$OUTFILE"

for N in 100000 500000 1000000 5000000 10000000; do
    echo -n "  n=$N ... "
    RAW=$(./Final/task1 "$N" "$T" "$TPB")

    MD5_S=$(echo  "$RAW" | sed -n '2p')
    MD5_O=$(echo  "$RAW" | sed -n '3p')
    MD5_G=$(echo  "$RAW" | sed -n '4p')
    SHA1_S=$(echo "$RAW" | sed -n '6p')
    SHA1_O=$(echo "$RAW" | sed -n '7p')
    SHA1_G=$(echo "$RAW" | sed -n '8p')
    S256_S=$(echo "$RAW" | sed -n '10p')
    S256_O=$(echo "$RAW" | sed -n '11p')
    S256_G=$(echo "$RAW" | sed -n '12p')

    echo "$N  $MD5_S  $MD5_O  $MD5_G  $SHA1_S  $SHA1_O  $SHA1_G  $S256_S  $S256_O  $S256_G" \
        >> "$OUTFILE"
    echo "done"
done

echo "=== Written to $OUTFILE ==="
cat "$OUTFILE"
