#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH -o logs/seq.log

./sobelf_seq images/original/TimelyHugeGnu.gif images/processed/TimelyHugeGnu-sobel-seq.gif

