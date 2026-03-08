#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw7_task1_cub
#SBATCH -o hw7_task1_cub_%j.out
#SBATCH -e hw7_task1_cub_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:05:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Usage: sbatch sbatch/task1_cub_hw7.sh <n>
#   n -- number of elements (default 1048576)

module load nvidia/cuda/13.0

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW07"

N=${1:-1048576}

nvcc task1_cub.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_cub

./task1_cub "$N"
