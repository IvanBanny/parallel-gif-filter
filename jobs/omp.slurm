#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o logs/omp.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./sobelf_omp images/original/TimelyHugeGnu.gif images/processed/TimelyHugeGnu-sobel-omp.gif

