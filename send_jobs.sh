#!/bin/bash

for job in jobs/*; do
    sbatch "$job" ${1:+"$1"}
done
