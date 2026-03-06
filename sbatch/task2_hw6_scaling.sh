#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw6_task2_scaling
#SBATCH -o hw6_task2_scaling_%j.out
#SBATCH -e hw6_task2_scaling_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Scaling study for HW06 Task 2 (Hillis-Steele inclusive scan).
# n = 2^10, 2^11, ..., 2^16 with threads_per_block=1024 per assignment spec.
# Output data file: HW06/scaling_task2.dat
# Columns: n  time_ms

module load nvidia/cuda/13.0

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW06"

nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

TPB=1024
OUT="scaling_task2.dat"
echo "# n time_ms (threads_per_block=$TPB)" > "$OUT"

for exp in 10 11 12 13 14 15 16; do
    n=$((1 << exp))
    time_ms=$(./task2 "$n" "$TPB" | tail -1)
    echo "$n $time_ms" >> "$OUT"
    echo "n=$n done: ${time_ms} ms"
done

echo "Scaling data written to $OUT"
