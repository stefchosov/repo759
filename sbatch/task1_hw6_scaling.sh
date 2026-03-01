#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J hw6_task1_scaling
#SBATCH -o hw6_task1_scaling_%j.out
#SBATCH -e hw6_task1_scaling_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=16G
#SBATCH -c 1

# Scaling study for HW06 Task 1 (cuBLAS matrix multiplication).
# Varies n over powers of 2 from 64 to 8192 and records timing.
# Output data file: HW06/scaling_task1.dat
# Columns: n  time_ms

cd $SLURM_SUBMIT_DIR

nvcc -O3 -std=c++14 \
    -o HW06/task1 \
    HW06/task1.cu HW06/mmul.cpp \
    -lcublas

OUT="HW06/scaling_task1.dat"
echo "# n time_ms" > "$OUT"

for n in 64 128 256 512 1024 2048 4096 8192; do
    # task1 prints: C[n*n-1]\ntime_ms
    time_ms=$(./HW06/task1 "$n" | tail -1)
    echo "$n $time_ms" >> "$OUT"
done

echo "Scaling data written to $OUT"
