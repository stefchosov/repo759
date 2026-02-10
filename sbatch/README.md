# HW03 SBATCH Scripts

This directory contains SLURM batch scripts for running HW03 CUDA programs on Euler.

## Quick Start

From this `sbatch` directory, submit jobs using:

```bash
sbatch task1.sbatch
sbatch task2.sbatch
sbatch task3.sbatch
sbatch task3_scaling.sbatch
```

## Scripts Description

### task1.sbatch
- Compiles and runs `task1.cu` (factorial computation)
- Output: `task1.out`, errors: `task1.err`
- Expected: Prints factorials 1! through 8!

### task2.sbatch
- Compiles and runs `task2.cu` (ax+y computation)
- Output: `task2.out`, errors: `task2.err`
- Expected: 16 space-separated integers

### task3.sbatch
- Compiles and runs `task3.cu` with test input (n=2^20)
- Output: `task3.out`, errors: `task3.err`
- Expected: execution time (ms), first element, last element

### task3_scaling.sbatch
- **Full scaling study** for Problem 3c
- Runs task3 for n = 2^10 through 2^29
- Tests both 512 and 16 threads per block
- Generates data files: `../HW03/scaling_512.dat` and `../HW03/scaling_16.dat`
- Output: `task3_scaling.out`, errors: `task3_scaling.err`
- Runtime: ~20-30 minutes

## Generating the Plot (Problem 3c)

After `task3_scaling.sbatch` completes:

```bash
# From the sbatch directory
python3 generate_plot.py
```

This creates `task3.pdf` in the HW03 directory (required for submission).

## Checking Job Status

```bash
squeue -u $USER          # View your jobs
cat task1.out            # View output
cat task1.err            # View errors
scancel <job_id>         # Cancel a job
```

## Notes

- All scripts request GPU resources (`--gres=gpu:1`)
- CUDA module 13.0.0 is loaded automatically
- Compilation flags match assignment requirements
- All programs compile and run from the HW03 directory
