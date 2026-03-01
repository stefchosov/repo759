#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J hw6_task2
#SBATCH -o hw6_task2_%j.out
#SBATCH -e hw6_task2_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:05:00
#SBATCH --mem=8G
#SBATCH -c 1

# Usage: sbatch sbatch/task2_hw6.sh <n> <threads_per_block>
#   n                 -- number of elements to scan (default 1048576)
#   threads_per_block -- threads per block (default 1024)
# Compiles HW06/task2 and runs a single timing measurement.
# Requirement: n <= threads_per_block * threads_per_block

cd $SLURM_SUBMIT_DIR

N=${1:-1048576}
TPB=${2:-1024}

nvcc -O3 -std=c++14 \
    -o HW06/task2 \
    HW06/task2.cu HW06/scan.cu

./HW06/task2 "$N" "$TPB"
