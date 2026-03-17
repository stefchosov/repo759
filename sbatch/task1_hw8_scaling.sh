#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw8_task1_scaling
#SBATCH -o hw8_task1_scaling_%j.out
#SBATCH -e hw8_task1_scaling_%j.err
#SBATCH -t 0-00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20

# Scaling study for HW08 Task 1: n=1024, t=1..20
# Output: HW08/scaling_task1.dat  (columns: t  time_ms)

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW08"

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

N=1024
OUT="scaling_task1.dat"
echo "# t time_ms (n=$N)" > "$OUT"

for t in $(seq 1 20); do
    time_ms=$(./task1 "$N" "$t" | tail -1)
    echo "$t $time_ms" >> "$OUT"
    echo "t=$t done: ${time_ms} ms"
done

echo "Data written to $OUT"
