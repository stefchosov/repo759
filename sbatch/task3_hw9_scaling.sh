#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw9_task3_scaling
#SBATCH -o hw9_task3_scaling_%j.out
#SBATCH -e hw9_task3_scaling_%j.err
#SBATCH -t 0-00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

# Scaling study for HW09 Task 3: MPI ping-pong, n=2^1..2^25
# Output: HW09/scaling_task3.dat  (columns: n  time_ms)

module load nvidia/nvhpc-openmpi3/24.5

cd "${SLURM_SUBMIT_DIR%/sbatch}/HW09"

mpicxx task3.cpp -Wall -O3 -o task3

OUT="scaling_task3.dat"
echo "# n time_ms (t0+t1)" > "$OUT"

for exp in $(seq 1 25); do
    n=$((1 << exp))
    time_ms=$(mpirun -n 2 ./task3 "$n")
    echo "$n $time_ms" >> "$OUT"
    echo "n=$n done: ${time_ms} ms"
done

echo "Data written to $OUT"
