#!/usr/bin/env bash
#SBATCH --job-name=task2_test
#SBATCH --partition=instruction
#SBATCH --time=00-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task2_test.out
#SBATCH --error=task2_test.err

# Load CUDA module
module load nvidia/cuda/13.0.0

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Navigate to HW04 directory
cd ../HW04

# Compile the program
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xlinker -lm -o task2

# Run with sample parameters: n=1024, R=16, threads=256
echo "Running task2 with n=1024, R=16, threads_per_block=256"
./task2 1024 16 256
