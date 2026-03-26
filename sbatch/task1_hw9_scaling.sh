#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw9_task1_scaling
#SBATCH -o hw9_task1_scaling_%j.out
#SBATCH -e hw9_task1_scaling_%j.err
#SBATCH -t 0-00:30:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10

# Scaling study for HW09 Task 1: n=5040000, t=1..10
# Runs cluster 10 times per (n,t) and records the average to smooth spikes.
# Output: HW09/scaling_task1.dat  (columns: t  avg_time_ms)

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW09"

g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

N=5040000
REPS=10
OUT="scaling_task1.dat"
echo "# t avg_time_ms (n=$N, averaged over $REPS runs)" > "$OUT"

for t in $(seq 1 10); do
    total=0
    for rep in $(seq 1 $REPS); do
        ms=$(./task1 "$N" "$t" | tail -1)
        total=$(awk -v a="$total" -v b="$ms" 'BEGIN {print a + b}')
    done
    avg=$(awk -v s="$total" -v r="$REPS" 'BEGIN {print s / r}')
    echo "$t $avg" >> "$OUT"
    echo "t=$t done: avg=${avg} ms"
done

echo "Data written to $OUT"
