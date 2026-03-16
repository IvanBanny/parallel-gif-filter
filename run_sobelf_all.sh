#!/bin/bash
# run_sobelf_all.sh

NODES=1
RANKS=4
THREADS=2
INPUT="images/original/1.gif"
OUTPUT="images/processed/1.gif"

while getopts "N:n:c:i:o:s:" opt; do
    case $opt in
        N) NODES=$OPTARG ;;
        n) RANKS=$OPTARG ;;
        c) THREADS=$OPTARG ;;
        i) INPUT=$OPTARG ;;
        o) OUTPUT=$OPTARG ;;
        *) echo "Usage: $0 -N nodes -n ranks -c threads -i input.gif -o output.gif -s strategy"; exit 1 ;;
    esac
done

echo "Running Sobel with:"
echo "  Nodes       = $NODES"
echo "  MPI ranks   = $RANKS"
echo "  OMP threads = $THREADS"
echo "  Input       = $INPUT"
echo "  Output      = $OUTPUT"

salloc -N$NODES -n$RANKS -c$THREADS
export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=true
export OMP_PLACES=cores 
mpirun -np $RANKS --bind-to none ./sobelf_all $INPUT $OUTPUT