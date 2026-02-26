#!/usr/bin/env zsh
#SBATCH --job-name=HW05Task1BD
#SBATCH --partition=instruction
#SBATCH --time=00-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task1_blockdim-%j.out
#SBATCH --error=task1_blockdim-%j.err

module load nvidia/cuda/13.0

cd $SLURM_SUBMIT_DIR/../HW05

# Compile (skip if already built by task1.sh)
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

# Find best block_dim for n = 2^14 = 16384
n=16384
OUTPUT="blockdim_n${n}.dat"
echo "# block_dim time_int_ms time_float_ms time_double_ms" > $OUTPUT

for block_dim in 8 16 32; do
    out=$(./task1 $n $block_dim)
    t_int=$(echo "$out"    | awk 'NR==3')
    t_float=$(echo "$out"  | awk 'NR==6')
    t_double=$(echo "$out" | awk 'NR==9')
    echo "$block_dim $t_int $t_float $t_double" >> $OUTPUT
    echo "block_dim=$block_dim done"
done

echo "Block dim study complete: $OUTPUT"
