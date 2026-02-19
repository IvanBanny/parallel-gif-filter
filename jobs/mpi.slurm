#!/bin/bash
#SBATCH -n 4
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH -o logs/mpi.log

mpirun ./sobelf_mpi images/original/TimelyHugeGnu.gif images/processed/TimelyHugeGnu-sobel-mpi.gif

