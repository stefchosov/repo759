#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw6_task1
#SBATCH -o hw6_task1_%j.out
#SBATCH -e hw6_task1_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:05:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Usage: sbatch sbatch/task1_hw6.sh <n>
#   n   -- matrix dimension (default 1024)
# Compiles HW06/task1 and runs a single timing measurement.

module load nvidia/cuda/13.0

cd "${SLURM_SUBMIT_DIR%/sbatch}"

N=${1:-1024}

nvcc HW06/task1.cu HW06/mmul.cpp \
    -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 \
    -lcublas -o HW06/task1

./HW06/task1 "$N"
