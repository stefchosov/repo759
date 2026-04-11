#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task1_final_gpu
#SBATCH -o Final/sbatch/logs/task1_final_gpu_%j.out
#SBATCH -t 0-00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# GPU scaling study for Task 1 — to be filled in when task1.cu is implemented.
# Will produce: Final/Data/scaling_task1_gpu.dat
#
# Columns (must match CPU dat columns for the merge in plot_final_task1.py):
#   n  md5_gpu_ms  sha1_gpu_ms  sha256_gpu_ms

set -e
cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"

echo "GPU task1 script not yet implemented — waiting on task1.cu"
exit 1
