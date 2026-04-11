#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task1_final
#SBATCH -o Final/sbatch/logs/task1_final_%j.out
#SBATCH -t 0-00:05:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G

# Single-run sanity check for Task 1 (serial + OpenMP CPU)
# Usage: sbatch Final/sbatch/task1_final.sh [n] [t]
#   n  — number of hashes per algorithm (default: 10000000)
#   t  — OpenMP thread count            (default: 20)
#
# Output (9 lines to .out file):
#   <md5_check>  <md5_serial_ms>  <md5_omp_ms>
#   <sha1_check> <sha1_serial_ms> <sha1_omp_ms>
#   <sha256_check> <sha256_serial_ms> <sha256_omp_ms>

set -e
cd "$SLURM_SUBMIT_DIR"

N=${1:-10000000}
T=${2:-20}

echo "=== Task 1 — CPU throughput (n=$N, t=$T) ==="
echo "Host: $(hostname)  Cores: $SLURM_CPUS_PER_TASK"

g++ -O3 -std=c++17 -fopenmp \
    -o Final/task1 \
    Final/Code/task1.cpp \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp

echo "--- raw output ---"
./Final/task1 "$N" "$T"
