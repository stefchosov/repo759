#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J hw6_task1
#SBATCH -o hw6_task1_%j.out
#SBATCH -e hw6_task1_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:05:00
#SBATCH --mem=16G
#SBATCH -c 1

# Usage: sbatch sbatch/task1_hw6.sh <n>
#   n   -- matrix dimension (default 1024)
# Compiles HW06/task1 and runs a single timing measurement.

cd $SLURM_SUBMIT_DIR

N=${1:-1024}

nvcc -O3 -std=c++14 \
    -o HW06/task1 \
    HW06/task1.cu HW06/mmul.cpp \
    -lcublas

./HW06/task1 "$N"
