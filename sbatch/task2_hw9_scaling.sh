#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw9_task2_scaling
#SBATCH -o hw9_task2_scaling_%j.out
#SBATCH -e hw9_task2_scaling_%j.err
#SBATCH -t 0-00:30:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10

# Scaling study for HW09 Task 2: n=10^6, t=1..10
# Compiles two binaries: with simd and without simd (NOSIMD).
# Runs each 10 times per t and records the average.
# Output:
#   HW09/scaling_task2_simd.dat    (columns: t  avg_time_ms)
#   HW09/scaling_task2_nosimd.dat  (columns: t  avg_time_ms)

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW09"

g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2_simd \
    -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec 2>&1 | grep -v "^$"

g++ task2.cpp montecarlo.cpp -DNOSIMD -Wall -O3 -std=c++17 -o task2_nosimd \
    -fopenmp -fno-tree-vectorize -march=native 2>&1 | grep -v "^$"

N=1000000
REPS=10

for variant in simd nosimd; do
    OUT="scaling_task2_${variant}.dat"
    echo "# t avg_time_ms (n=$N, $variant, averaged over $REPS runs)" > "$OUT"

    for t in $(seq 1 10); do
        total=0
        for rep in $(seq 1 $REPS); do
            ms=$(./task2_${variant} "$N" "$t" | tail -1)
            total=$(awk -v a="$total" -v b="$ms" 'BEGIN {print a + b}')
        done
        avg=$(awk -v s="$total" -v r="$REPS" 'BEGIN {print s / r}')
        echo "$t $avg" >> "$OUT"
        echo "$variant t=$t done: avg=${avg} ms"
    done

    echo "Data written to $OUT"
done
