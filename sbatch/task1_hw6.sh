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

# Usage: sbatch sbatch/task1_hw6.sh <n> <n_tests>
#   n        -- matrix dimension (default 1024)
#   n_tests  -- number of mmul calls to average over (default 10)

module load nvidia/cuda/13.0

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW06"

N=${1:-1024}
N_TESTS=${2:-10}

nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o task1

./task1 "$N" "$N_TESTS"
