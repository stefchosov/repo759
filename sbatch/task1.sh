#!/usr/bin/env zsh
#SBATCH --job-name=HW05Task1
#SBATCH --partition=instruction
#SBATCH --time=00-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task1-%j.out
#SBATCH --error=task1-%j.err

module load nvidia/cuda/13.0

cd $SLURM_SUBMIT_DIR/../HW05

# Compile
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

BLOCK_DIM=16
OUTPUT="scaling_task1_${BLOCK_DIM}.dat"
echo "# n time_int_ms time_float_ms time_double_ms" > $OUTPUT

for exp in 5 6 7 8 9 10 11 12 13 14; do
    n=$((1 << exp))
    out=$(./task1 $n $BLOCK_DIM)
    t_int=$(echo "$out"    | awk 'NR==3')
    t_float=$(echo "$out"  | awk 'NR==6')
    t_double=$(echo "$out" | awk 'NR==9')
    echo "$n $t_int $t_float $t_double" >> $OUTPUT
    echo "n=$n done"
done

echo "Scaling study complete: $OUTPUT"
