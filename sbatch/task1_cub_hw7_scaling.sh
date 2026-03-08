#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw7_task1_cub_scaling
#SBATCH -o hw7_task1_cub_scaling_%j.out
#SBATCH -e hw7_task1_cub_scaling_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Scaling study: cub::DeviceReduce::Sum for n = 2^10 .. 2^20
# Output: HW07/scaling_task1_cub.dat  (columns: n  time_ms)

module load nvidia/cuda/13.0

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW07"

nvcc task1_cub.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_cub

OUT="scaling_task1_cub.dat"
echo "# n time_ms" > "$OUT"

for exp in 10 11 12 13 14 15 16 17 18 19 20; do
    n=$((1 << exp))
    time_ms=$(./task1_cub "$n" | tail -1)
    echo "$n $time_ms" >> "$OUT"
    echo "n=$n done: ${time_ms} ms"
done

echo "Data written to $OUT"
