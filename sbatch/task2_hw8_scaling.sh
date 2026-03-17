#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw8_task2_scaling
#SBATCH -o hw8_task2_scaling_%j.out
#SBATCH -e hw8_task2_scaling_%j.err
#SBATCH -t 0-00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20

# Scaling study for HW08 Task 2: n=1024, t=1..20
# Output: HW08/scaling_task2.dat  (columns: t  time_ms)

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW08"

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

N=1024
OUT="scaling_task2.dat"
echo "# t time_ms (n=$N)" > "$OUT"

for t in $(seq 1 20); do
    time_ms=$(./task2 "$N" "$t" | tail -1)
    echo "$t $time_ms" >> "$OUT"
    echo "t=$t done: ${time_ms} ms"
done

echo "Data written to $OUT"
