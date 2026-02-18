#!/bin/bash
#SBATCH -n 4
#SBATCH -c 8
#SBATCH -o job.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun ./sobelf images/original/TimelyHugeGnu.gif images/processed/TimelyHugeGnu-sobel.gif

