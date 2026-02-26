#!/usr/bin/env zsh
#SBATCH --job-name=HW02mmul1
#SBATCH --partition=instruction
#SBATCH --time=00-03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=hw02_mmul1-%j.out
#SBATCH --error=hw02_mmul1-%j.err

cd $SLURM_SUBMIT_DIR/../HW02

g++ test_mmul1.cpp matmul.cpp -O3 -Wall -std=c++17 -o test_mmul1

echo "Running mmul1 at n=16384"
./test_mmul1 16384
