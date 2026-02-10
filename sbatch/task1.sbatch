#!/usr/bin/env bash
#SBATCH --job-name=HW03_task1
#SBATCH --partition=instruction
#SBATCH --time=00-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task1.out
#SBATCH --error=task1.err

# Load CUDA module
module load nvidia/cuda/13.0.0

# Navigate to HW03 directory
cd $SLURM_SUBMIT_DIR/../HW03

# Compile
nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Run
./task1
