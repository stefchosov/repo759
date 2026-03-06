#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw6_task1_scaling
#SBATCH -o hw6_task1_scaling_%j.out
#SBATCH -e hw6_task1_scaling_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Scaling study for HW06 Task 1 (cuBLAS matrix multiplication).
# Varies n over powers of 2 from 64 to 8192 and records timing.
# Output data file: HW06/scaling_task1.dat
# Columns: n  time_ms

module load nvidia/cuda/13.0

cd $SLURM_SUBMIT_DIR

nvcc HW06/task1.cu HW06/mmul.cpp \
    -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 \
    -lcublas -o HW06/task1

OUT="HW06/scaling_task1.dat"
echo "# n time_ms" > "$OUT"

for exp in 6 7 8 9 10 11 12 13; do
    n=$((1 << exp))
    time_ms=$(./HW06/task1 "$n" | tail -1)
    echo "$n $time_ms" >> "$OUT"
    echo "n=$n done"
done

echo "Scaling data written to $OUT"
