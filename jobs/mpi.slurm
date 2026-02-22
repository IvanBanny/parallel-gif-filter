#!/bin/bash
#SBATCH -n 4
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH -o logs/mpi.log

IN_DIR="images/original"
OUT_DIR="images/processed"
GIF=${1:-walle}
SCRIPT="mpi"

mpirun ./sobelf_$SCRIPT $IN_DIR/$GIF.gif $OUT_DIR/$GIF-$SCRIPT.gif
