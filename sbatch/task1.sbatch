#!/usr/bin/env bash
#SBATCH --job-name=task1_test
#SBATCH --partition=instruction
#SBATCH --time=00-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task1_test.out
#SBATCH --error=task1_test.err

# Load CUDA module
module load nvidia/cuda/13.0.0

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Navigate to HW04 directory
cd ../HW04

# Compile the program
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xlinker -lm -o task1

# Run with sample parameters: n=64, threads=256
echo "Running task1 with n=64, threads_per_block=256"
./task1 64 256
