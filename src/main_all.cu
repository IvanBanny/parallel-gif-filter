
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

#include "sobel_cuda.h"
#include "sobel_hyb.h"
#include "sobel_mpi.h"
#include "sobel_omp.h"

enum STRATEGY
{
    MPI,
    OMP,
    HYB,
    CUDA
};

/*
salloc -N1 -n1 -c8
export OMP_NUM_THREADS=8 
export OMP_PROC_BIND=true 
export OMP_PLACES=cores 
mpirun -np 1 --bind-to none ./sobelf_all images/original/1.gif images/processed/1.gif 0

*/

int main(int argc, char** argv)
{
    int rank = 0;
    int n_ranks = 0;
    int n_images = 0;

    animated_gif* image = NULL;
    
    double t_start = 0.0, t_end = 0.0, duration = 0.0;

    const char* input_filename = NULL;
    const char* output_filename = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (argc < 3)
    {
        if (rank == 0) 
        {
            fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    input_filename = argv[1];
    output_filename = argv[2];

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    // data loading
    image = load_input_on_root(rank, input_filename, &n_images);

    t_end = MPI_Wtime();
    duration = t_end - t_start;
    if (rank == 0)
    {
        printf("GIF loaded from file %s with %d image(s) in %lf s\n", input_filename, n_images, duration);
    }

    cuda_warmup();

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    STRATEGY strategy = OMP;
    // CHOOSING STRATEGY

    switch(*argv[3])
    {
        case '0':
            strategy = OMP;
            break;
        case '1':
            strategy = MPI;
            break;
        case '2':
            strategy = HYB;
            break;
        case '3':
            strategy = CUDA;
            break;
        default:
            printf("wtf\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // CHOOSING STRATEGY - END
    t_end = MPI_Wtime();
    duration = t_end - t_start;
    if (rank == 0) 
    {
        printf("Choosing strategy done in %lf s\n", duration);
        switch(strategy)
        {
            case OMP:
                printf("strategy : omp\n");
                break;
            case MPI:
                printf("strategy : mpi\n");
                break;
            case HYB:
                printf("strategy : hyb\n");
                break;
            case CUDA:
                printf("strategy : cuda\n");
                break;
            default:
                printf("not a valid strategy tf !?\n");
        }
    }
    double choosing_strategy_duration = duration;

    if(strategy == CUDA)
    {
        cuda_warmup();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    // FILTER

    switch(strategy)
    {
        case OMP:
            if(rank==0) { sobel_omp(image); }
            break;
        case MPI:
            sobel_mpi(image, n_images, rank, n_ranks);
            break;
        case HYB:
            sobel_hyb(image, n_images, rank, n_ranks);
            break;
        case CUDA:
            if(rank == 0) { sobel_cuda(image); }
            break;
        default:
            printf("not a valid strategy tf !?\n");

    }

    // FILTER - END
    t_end = MPI_Wtime();
    duration = t_end - t_start;
    if (rank == 0) 
    {
        printf("SOBEL done in %lf s\n", duration);
        printf("TOTAL (choosing strategy + filter) = %lf s\n", duration + choosing_strategy_duration);
    }
    
    // Output gif
#if !defined(SKIP_EXPORT)

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    if (rank == 0 && store_pixels((char*) output_filename, image)) 
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    t_end = MPI_Wtime();
    duration = t_end - t_start;
    if (rank == 0) 
    {
        printf("Export done in %lf s in file %s\n", duration, output_filename);
    }

#endif

    MPI_Finalize();

    return 0;
}