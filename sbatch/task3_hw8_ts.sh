#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw8_task3_ts
#SBATCH -o hw8_task3_ts_%j.out
#SBATCH -e hw8_task3_ts_%j.err
#SBATCH -t 0-00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20

# Scaling study for HW08 Task 3: n=10^6, t=8, ts=2^1..2^10
# Output: HW08/scaling_task3_ts.dat  (columns: ts  time_ms)

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW08"

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

N=1000000
T=8
OUT="scaling_task3_ts.dat"
echo "# ts time_ms (n=$N, t=$T)" > "$OUT"

for exp in 1 2 3 4 5 6 7 8 9 10; do
    ts=$((1 << exp))
    time_ms=$(./task3 "$N" "$T" "$ts" | tail -1)
    echo "$ts $time_ms" >> "$OUT"
    echo "ts=$ts done: ${time_ms} ms"
done

echo "Data written to $OUT"
