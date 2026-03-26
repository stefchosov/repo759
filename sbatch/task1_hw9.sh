#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw9_task1
#SBATCH -o hw9_task1_%j.out
#SBATCH -e hw9_task1_%j.err
#SBATCH -t 0-00:10:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10

# Usage: sbatch sbatch/task1_hw9.sh [n] [t]
#   n -- array length (default 5040000, must be multiple of 2*t)
#   t -- threads (default 8)

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW09"

g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

N=${1:-5040000}
T=${2:-8}

./task1 "$N" "$T"
