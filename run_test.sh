#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed

mkdir -p "$OUTPUT_DIR/mpi" "$OUTPUT_DIR/omp" "$OUTPUT_DIR/hyb" "$OUTPUT_DIR/cuda"

for i in $INPUT_DIR/*gif ; do
    DEST=$OUTPUT_DIR/mpi/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST"

    salloc -N 1 -n 4 mpirun ./sobelf_mpi $i $DEST

    DEST=$OUTPUT_DIR/omp/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST"

    salloc -n 1 ./sobelf_omp $i $DEST

    DEST=$OUTPUT_DIR/hyb/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST"

    salloc -N 1 -n 4 ./sobelf_hyb $i $DEST

    DEST=$OUTPUT_DIR/cuda/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST"

    salloc -n 1 -N 1 ./sobelf_cuda $i $DEST
done