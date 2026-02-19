#!/usr/bin/env bash
#SBATCH --job-name=task2_scaling
#SBATCH --partition=instruction
#SBATCH --time=00-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task2_scaling.out
#SBATCH --error=task2_scaling.err

# Load CUDA module
module load nvidia/cuda/13.0.0

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Navigate to HW04 directory
cd ../HW04

# Compile the program
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xlinker -lm -o task2

# Output files
OUTPUT_1024="scaling_task2_1024.dat"
OUTPUT_512="scaling_task2_512.dat"

# Clear output files if they exist
> $OUTPUT_1024
> $OUTPUT_512

echo "Running scaling study for task2..."

# Fixed R value
R=128

# Test with 1024 threads/block
echo "# n time_ms" > $OUTPUT_1024
for power in {10..29}; do
    n=$((2**power))
    echo "Testing n=$n, R=$R with 1024 threads/block"
    result=$(./task2 $n $R 1024)
    time=$(echo "$result" | tail -1)
    echo "$n $time" >> $OUTPUT_1024
done

# Test with 512 threads/block
echo "# n time_ms" > $OUTPUT_512
for power in {10..29}; do
    n=$((2**power))
    echo "Testing n=$n, R=$R with 512 threads/block"
    result=$(./task2 $n $R 512)
    time=$(echo "$result" | tail -1)
    echo "$n $time" >> $OUTPUT_512
done

echo "Scaling study complete!"
echo "Output files: $OUTPUT_1024, $OUTPUT_512"
