#!/bin/bash
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o logs/hyb.log

IN_DIR="images/original"
OUT_DIR="images/processed"
GIF=${1:-walle}
SCRIPT="hyb"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK --bind-to core \
    ./sobelf_$SCRIPT $IN_DIR/$GIF.gif $OUT_DIR/$GIF-$SCRIPT.gif
