#!/usr/bin/env bash
#SBATCH --job-name=HW03_task3
#SBATCH --partition=instruction
#SBATCH --time=00-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task3.out
#SBATCH --error=task3.err

# Load CUDA module
module load nvidia/cuda/13.0.0

# Navigate to HW03 directory
cd $SLURM_SUBMIT_DIR/../HW03

# Compile
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

# Run with test size (2^20)
echo "Testing with n=1048576 (2^20):"
./task3 1048576
