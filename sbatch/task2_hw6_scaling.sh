#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J hw6_task2_scaling
#SBATCH -o hw6_task2_scaling_%j.out
#SBATCH -e hw6_task2_scaling_%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Scaling study for HW06 Task 2 (Hillis-Steele inclusive scan).
# Part A: fixes threads_per_block=1024, varies n from 32 to 1048576.
# Part B: fixes n=1048576, varies threads_per_block in {32, 64, 128, 256, 512, 1024}.
# Output data files:
#   HW06/scaling_task2_n.dat    -- columns: n  time_ms
#   HW06/scaling_task2_tpb.dat  -- columns: tpb  time_ms

module load nvidia/cuda/13.0

cd $SLURM_SUBMIT_DIR

nvcc HW06/task2.cu HW06/scan.cu \
    -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 \
    -o HW06/task2

# ---------- Part A: vary n with fixed tpb=1024 ----------
TPB=1024
OUT_N="HW06/scaling_task2_n.dat"
echo "# n time_ms (threads_per_block=$TPB)" > "$OUT_N"

for exp in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    n=$((1 << exp))
    time_ms=$(./HW06/task2 "$n" "$TPB" | tail -1)
    echo "$n $time_ms" >> "$OUT_N"
    echo "n=$n done"
done

echo "n-scaling data written to $OUT_N"

# ---------- Part B: vary tpb with fixed n=1048576 ----------
N=1048576
OUT_TPB="HW06/scaling_task2_tpb.dat"
echo "# tpb time_ms (n=$N)" > "$OUT_TPB"

for tpb in 32 64 128 256 512 1024; do
    time_ms=$(./HW06/task2 "$N" "$tpb" | tail -1)
    echo "$tpb $time_ms" >> "$OUT_TPB"
    echo "tpb=$tpb done"
done

echo "tpb-scaling data written to $OUT_TPB"
