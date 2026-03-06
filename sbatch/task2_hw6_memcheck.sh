#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw6_task2_memcheck
#SBATCH -o hw6_task2_memcheck_%j.out
#SBATCH -e hw6_task2_memcheck_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:05:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Runs cuda-memcheck on task2 with n=2^10=1024 and threads_per_block=1024
# per assignment spec (Problem 2a). Output goes to the .out file above.

module load nvidia/cuda/13.0

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW06"

nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

compute-sanitizer ./task2 1024 1024
