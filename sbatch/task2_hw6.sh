#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw6_task2
#SBATCH -o hw6_task2_%j.out
#SBATCH -e hw6_task2_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:05:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Usage: sbatch sbatch/task2_hw6.sh <n> <threads_per_block>
#   n                 -- number of elements to scan (default 1024)
#   threads_per_block -- threads per block (default 1024)

module load nvidia/cuda/13.0

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW06"

N=${1:-1024}
TPB=${2:-1024}

nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

./task2 "$N" "$TPB"
