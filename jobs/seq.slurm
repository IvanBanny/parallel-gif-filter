#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH -o logs/seq.log

IN_DIR="images/original"
OUT_DIR="images/processed"
GIF=${1:-walle}
SCRIPT="seq"

./sobelf_$SCRIPT $IN_DIR/$GIF.gif $OUT_DIR/$GIF-$SCRIPT.gif
