#!/bin/bash
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o logs/hybrid.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun ./sobelf_hybrid images/original/TimelyHugeGnu.gif images/processed/TimelyHugeGnu-sobel-hybrid.gif

