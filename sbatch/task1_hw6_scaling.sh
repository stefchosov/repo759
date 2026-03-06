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
# n = 2^5, 2^6, ..., 2^11 per assignment spec.
# Output data file: HW06/scaling_task1.dat
# Columns: n  avg_time_ms

module load nvidia/cuda/13.0

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW06"

nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o task1

N_TESTS=10

OUT="scaling_task1.dat"
echo "# n avg_time_ms" > "$OUT"

for exp in 5 6 7 8 9 10 11; do
    n=$((1 << exp))
    avg_ms=$(./task1 "$n" "$N_TESTS")
    echo "$n $avg_ms" >> "$OUT"
    echo "n=$n done: ${avg_ms} ms"
done

echo "Scaling data written to $OUT"
