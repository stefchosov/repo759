#!/usr/bin/env zsh
#SBATCH --job-name=HW05Task2
#SBATCH --partition=instruction
#SBATCH --time=00-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task2-%j.out
#SBATCH --error=task2-%j.err

module load nvidia/cuda/13.0

cd $SLURM_SUBMIT_DIR/../HW05

# Compile
nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

OUTPUT="scaling_task2.dat"
echo "# N time_tpb1024_ms time_tpb256_ms" > $OUTPUT

for exp in $(seq 10 30); do
    N=$((1 << exp))

    out_1024=$(./task2 $N 1024)
    t_1024=$(echo "$out_1024" | awk 'NR==2')

    out_256=$(./task2 $N 256)
    t_256=$(echo "$out_256" | awk 'NR==2')

    echo "$N $t_1024 $t_256" >> $OUTPUT
    echo "N=$N done"
done

echo "Scaling study complete: $OUTPUT"
