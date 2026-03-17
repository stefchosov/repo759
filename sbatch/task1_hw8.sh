#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw8_task1
#SBATCH -o hw8_task1_%j.out
#SBATCH -e hw8_task1_%j.err
#SBATCH -t 0-00:10:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20

# Usage: sbatch sbatch/task1_hw8.sh <n> <t>
#   n -- matrix dimension (default 1024)
#   t -- threads (default 8)
# No GPU, no module load required.

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW08"

N=${1:-1024}
T=${2:-8}

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

./task1 "$N" "$T"
