#!/usr/bin/env bash
#SBATCH --job-name=task1_scaling
#SBATCH --partition=instruction
#SBATCH --time=00-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task1_scaling.out
#SBATCH --error=task1_scaling.err

# Load CUDA module
module load nvidia/cuda/13.0.0

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Navigate to HW04 directory
cd ../HW04

# Compile the program
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xlinker -lm -o task1

# Output files
OUTPUT_1024="scaling_task1_1024.dat"
OUTPUT_512="scaling_task1_512.dat"

# Clear output files if they exist
> $OUTPUT_1024
> $OUTPUT_512

echo "Running scaling study for task1..."

# Test with 1024 threads/block
echo "# n time_ms" > $OUTPUT_1024
for power in {5..14}; do
    n=$((2**power))
    echo "Testing n=$n with 1024 threads/block"
    result=$(./task1 $n 1024)
    time=$(echo "$result" | tail -1)
    echo "$n $time" >> $OUTPUT_1024
done

# Test with 512 threads/block
echo "# n time_ms" > $OUTPUT_512
for power in {5..14}; do
    n=$((2**power))
    echo "Testing n=$n with 512 threads/block"
    result=$(./task1 $n 512)
    time=$(echo "$result" | tail -1)
    echo "$n $time" >> $OUTPUT_512
done

echo "Scaling study complete!"
echo "Output files: $OUTPUT_1024, $OUTPUT_512"
