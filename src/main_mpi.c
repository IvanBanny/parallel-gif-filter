/*
 * INF560
 *
 * Image Filtering Project
 * MPI Implementation
 */

#include "sobel_mpi.h"

// ==_MAIN == //

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

    // Input gif on root and sending dimensions to all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    image = load_input_on_root(rank, input_filename, &n_images);

    t_end = MPI_Wtime();
    duration = t_end - t_start;
    if (rank == 0)
    {
        printf("GIF loaded from file %s with %d image(s) in %lf s\n", input_filename, n_images, duration);
    }

    // Actual work
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    sobel_mpi(image, n_images, rank, n_ranks);

    t_end = MPI_Wtime();
    duration = t_end - t_start;

    if (rank == 0) 
    {
        printf("SOBEL done in %lf s\n", duration);
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
