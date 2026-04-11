#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task1_final_scaling
#SBATCH -o Final/sbatch/logs/task1_final_scaling_%j.out
#SBATCH -t 0-00:30:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G

# Scaling study for Task 1 — varies n, fixed t=20 threads.
# Produces: Final/Data/scaling_task1_cpu.dat
#
# Columns:
#   n  md5_serial_ms  md5_omp_ms  sha1_serial_ms  sha1_omp_ms  sha256_serial_ms  sha256_omp_ms
#
# GPU columns will be added later via task1_final_gpu_scaling.sh and
# merged by the plot script.

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

T=20
OUTFILE=Final/Data/scaling_task1_cpu.dat

echo "=== Task 1 scaling (t=$T) — $(date) ==="
echo "Host: $(hostname)"

# Compile once
g++ -O3 -std=c++17 -fopenmp \
    -o Final/task1 \
    Final/Code/task1.cpp \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp

# Write header
echo "# n  md5_serial_ms  md5_omp_ms  sha1_serial_ms  sha1_omp_ms  sha256_serial_ms  sha256_omp_ms" \
    > "$OUTFILE"

for N in 100000 500000 1000000 5000000 10000000 50000000 100000000; do
    echo -n "  n=$N ... "

    RAW=$(./Final/task1 "$N" "$T")

    MD5_SERIAL=$(echo  "$RAW" | sed -n '2p')
    MD5_OMP=$(echo     "$RAW" | sed -n '3p')
    SHA1_SERIAL=$(echo "$RAW" | sed -n '5p')
    SHA1_OMP=$(echo    "$RAW" | sed -n '6p')
    S256_SERIAL=$(echo "$RAW" | sed -n '8p')
    S256_OMP=$(echo    "$RAW" | sed -n '9p')

    echo "$N  $MD5_SERIAL  $MD5_OMP  $SHA1_SERIAL  $SHA1_OMP  $S256_SERIAL  $S256_OMP" \
        >> "$OUTFILE"
    echo "done"
done

echo "=== Written to $OUTFILE ==="
cat "$OUTFILE"
