#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw7_task3
#SBATCH -o hw7_task3_%j.out
#SBATCH -e hw7_task3_%j.err
#SBATCH -t 0-00:05:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

# No GPU needed for this task — note absence of --gres=gpu:1
# Compile with g++ and run the OpenMP task3 binary.

module load nvidia/cuda/13.0

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW07"

g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

./task3
