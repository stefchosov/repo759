#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw8_task3_t
#SBATCH -o hw8_task3_t_%j.out
#SBATCH -e hw8_task3_t_%j.err
#SBATCH -t 0-00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20

# Scaling study for HW08 Task 3: n=10^6, t=1..20, best ts from ts study.
# Usage: sbatch sbatch/task3_hw8_t.sh <best_ts>
#   best_ts -- threshold that gave best performance in ts study (default 512)
# Output: HW08/scaling_task3_t.dat  (columns: t  time_ms)

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW08"

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

N=1000000
TS=${1:-512}
OUT="scaling_task3_t.dat"
echo "# t time_ms (n=$N, ts=$TS)" > "$OUT"

for t in $(seq 1 20); do
    time_ms=$(./task3 "$N" "$t" "$TS" | tail -1)
    echo "$t $time_ms" >> "$OUT"
    echo "t=$t done: ${time_ms} ms"
done

echo "Data written to $OUT"
