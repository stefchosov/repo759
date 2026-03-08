#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw7_task2_scaling
#SBATCH -o hw7_task2_scaling_%j.out
#SBATCH -e hw7_task2_scaling_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Scaling study: count() for n = 2^5 .. 2^20
# Output: HW07/scaling_task2.dat  (columns: n  time_ms)

module load nvidia/cuda/13.0

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW07"

nvcc task2.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

OUT="scaling_task2.dat"
echo "# n time_ms" > "$OUT"

for exp in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    n=$((1 << exp))
    time_ms=$(./task2 "$n" | tail -1)
    echo "$n $time_ms" >> "$OUT"
    echo "n=$n done: ${time_ms} ms"
done

echo "Data written to $OUT"
