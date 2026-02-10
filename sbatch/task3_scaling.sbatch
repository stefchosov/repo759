#!/usr/bin/env bash
#SBATCH --job-name=HW03_task3_scaling
#SBATCH --partition=instruction
#SBATCH --time=00-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task3_scaling.out
#SBATCH --error=task3_scaling.err

# Load CUDA module
module load nvidia/cuda/13.0.0

# Navigate to HW03 directory
cd $SLURM_SUBMIT_DIR/../HW03

# Compile with 512 threads per block (default)
echo "Compiling task3 with 512 threads per block..."
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3_512

# Create a version with 16 threads per block
# First, create a modified task3.cu with 16 threads
sed 's/const int threadsPerBlock = 512;/const int threadsPerBlock = 16;/' task3.cu > task3_16.cu

echo "Compiling task3 with 16 threads per block..."
nvcc task3_16.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3_16

# Output file for results
RESULTS_512="scaling_512.dat"
RESULTS_16="scaling_16.dat"

# Clear previous results
> $RESULTS_512
> $RESULTS_16

echo "Running scaling study from 2^10 to 2^29..."
echo "n time_ms" > $RESULTS_512
echo "n time_ms" > $RESULTS_16

# Run for each power of 2 from 10 to 29
for i in {10..29}; do
    n=$((2**i))
    echo "Testing n = 2^$i = $n"

    # Run with 512 threads per block
    output_512=$(./task3_512 $n)
    time_512=$(echo "$output_512" | head -n 1)
    echo "$n $time_512" >> $RESULTS_512

    # Run with 16 threads per block
    output_16=$(./task3_16 $n)
    time_16=$(echo "$output_16" | head -n 1)
    echo "$n $time_16" >> $RESULTS_16

    echo "  512 threads/block: $time_512 ms"
    echo "  16 threads/block: $time_16 ms"
done

echo ""
echo "Results saved to:"
echo "  - $RESULTS_512 (512 threads per block)"
echo "  - $RESULTS_16 (16 threads per block)"
echo ""
echo "You can now generate the plot using these data files."
