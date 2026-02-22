#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o logs/omp.log

IN_DIR="images/original"
OUT_DIR="images/processed"
GIF=${1:-walle}
SCRIPT="omp"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./sobelf_$SCRIPT $IN_DIR/$GIF.gif $OUT_DIR/$GIF-$SCRIPT.gif
