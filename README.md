# Parallel Sobel Edge Detection on Animated GIFs

**INF560 - Parallel Computing** - Ecole Polytechnique - March 2026 - *Ivan Banny & Toan Lopez*

Applies a three-stage filter pipeline - grayscale, iterative box blur, and Sobel edge detection - to animated GIFs. Parallelized with **OpenMP**, **MPI**, **CUDA**, and a **unified version** that selects the best strategy at runtime via a trained decision tree.

## Dependencies

- **C compiler with OpenMP** (GCC via `mpicc`)
- **MPI** (e.g. OpenMPI)
- **CUDA Toolkit** (`nvcc`)
- **GNU Make**
- **SLURM** *(optional, for cluster job submission)*

## Build

```bash
make                  # all targets
make sobelf_seq       # sequential baseline
make sobelf_omp       # OpenMP
make sobelf_mpi       # MPI (adaptive frame/stripe modes)
make sobelf_hyb       # MPI + OpenMP hybrid
make sobelf_cuda      # CUDA
make sobelf_all       # unified (decision tree selects strategy)
make clean
```

## Run

All binaries: `./sobelf_<variant> input.gif output.gif`

```bash
# Sequential
./sobelf_seq images/original/walle.gif output.gif

# OpenMP (set thread count)
OMP_NUM_THREADS=8 ./sobelf_omp images/original/walle.gif output.gif

# MPI (auto-selects frame vs. stripe distribution)
mpirun -np 4 ./sobelf_mpi images/original/walle.gif output.gif

# MPI + OpenMP hybrid
OMP_NUM_THREADS=4 mpirun -np 4 --bind-to none ./sobelf_hyb images/original/walle.gif output.gif

# CUDA
./sobelf_cuda images/original/walle.gif output.gif

# Unified (decision tree picks best strategy)
OMP_NUM_THREADS=4 mpirun -np 4 --bind-to none ./sobelf_all images/original/walle.gif output.gif
```

### SLURM

```bash
sbatch jobs/mpi.slurm                    # default image (walle)
sbatch jobs/cuda.slurm Mandelbrot-large  # specific image (no .gif extension)
./send_jobs.sh                           # submit all job scripts
```

## Testing

```bash
./run_test.sh              # process all 18 test GIFs through all implementations
./clean_test.sh            # remove processed outputs and logs
```

Outputs go to `images/processed/{mpi,omp,hyb,cuda}/`. Compare visually to verify correctness.

## Project Structure

```
src/          main_seq.c, main_omp.c, main_mpi.c, main_hyb.c, main_cuda.cu, main_all.cu
              gif_io.c + bundled giflib sources
include/      gif_io.h, sobel_{omp,mpi,hyb,cuda}.h, giflib and CUDA helper headers
images/       original/ (18 test GIFs)
jobs/         SLURM scripts (seq, omp, mpi, hyb, cuda)
```