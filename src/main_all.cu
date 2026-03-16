
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

enum Strategy
{
    MPI,
    OMP,
    HYB,
    CUDA
};

static int get_number_of_nodes() 
{
    char* nodes_str = getenv("SLURM_JOB_NUM_NODES");
    if (nodes_str) 
    {
        printf("number of nodes : %i\n", atoi(nodes_str));
        return atoi(nodes_str);
    } 
    else 
    {
        printf("didnt find number of nodes\n");
        return 1;
    }
}


static Strategy strategy_decision_tree(int width, int height, int pixels_per_frame, int total_pixels, int threads, int nodes, int ranks, int cuda_available)
{
    if (cuda_available == 0.0) 
    {
        if (threads <= 3) 
        {
            if (width <= 115) 
            {
                return OMP; // replaced SEQ with OMP for now
            } 
            else 
            {
                if (nodes <= 1) 
                {
                    return MPI;
                } 
                else
                {
                    if (ranks <= 3) 
                    {
                        return MPI;
                    }
                    else
                    {
                        return OMP;
                    }
                }
            }
        } 
        else
        {
            if (total_pixels <= 61700)
            {
                if (width <= 115)
                {
                    return OMP; // replaced SEQ with OMP for now
                }
                else
                {
                    if (threads <= 6)
                    {
                        return OMP;
                    }
                    else 
                    {
                        return OMP; // replaced SEQ with OMP for now
                    }
                }
            }
            else
            {
                return OMP;
            }
        }
    } 
    else 
    {
        if (total_pixels <= 1854500) 
        {
            if (pixels_per_frame <= 20900) 
            {
                return OMP; // replaced SEQ with OMP for now
            }
            else 
            {
                if (threads <= 3) 
                {
                    if (total_pixels <= 491625)
                    {
                        return MPI;
                    } 
                    else 
                    {
                        return CUDA;
                    }
                } 
                else 
                {
                    if (total_pixels <= 491625) 
                    {
                        return OMP;
                    } 
                    else 
                    {
                        return OMP;
                    }
                }
            }
        }
        else 
        {
            if (total_pixels <= 8100000) 
            {
                if (height <= 490) 
                {
                    return CUDA;
                } 
                else 
                {
                    if (threads <= 3) 
                    {
                        return CUDA;
                    } 
                    else 
                    {
                        return OMP;
                    }
                }
            } 
            else 
            {
                return CUDA;
            }
        }
    }
}


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

    Strategy strategy = OMP;
    // CHOOSING STRATEGY
    
    if(rank == 0)
    {
        int width = image->width[0];
        int height = image->height[0];
        int pixels_per_frame = width * height;
        int total_pixels = pixels_per_frame * n_images;
        int threads = omp_get_max_threads();
        int nodes = get_number_of_nodes();
        int ranks = n_ranks;
        
        int device_count = 0;
        checkCudaErrors(cudaGetDeviceCount(&device_count));
        int cuda_available_flag = device_count > 0 ? 1 : 0;

        strategy = strategy_decision_tree(width, height, pixels_per_frame, total_pixels, threads, nodes, ranks, cuda_available_flag);
    }
    MPI_Bcast(&strategy, 1, MPI_INT, 0, MPI_COMM_WORLD);

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