#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=FirstSlurm
#SBATCH --cpus-per-task=2
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err

hostname
