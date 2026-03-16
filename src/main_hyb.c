/*
 * INF560
 *
 * Image Filtering Project
 * MPI Implementation
 */

#include "sobel_hyb.h"

int main(int argc, char **argv) {
    int rank, n_ranks;
    char *input_filename, *output_filename;
    animated_gif *image = NULL;
    double t_start, t_end, duration;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    /* Check command-line arguments */
    if (rank == 0 && argc < 3) {
        fprintf(stderr, "Usage: %s input.gif output.gif \n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    input_filename = argv[1];
    output_filename = argv[2];

    /* IMPORT Timer start */
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    int n_images;
    image = load_input_on_root_hyb(rank, input_filename, &n_images);

    /* IMPORT Timer stop */
    t_end = MPI_Wtime();

    duration = t_end - t_start;

    if (rank == 0) {
        printf("GIF loaded from file %s with %d image(s) in %lf s\n",
               input_filename, n_images, duration);
    }

    /* FILTER Timer start */
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    sobel_hyb(image, n_images, rank, n_ranks);

    /* FILTER Timer stop */
    t_end = MPI_Wtime();

    duration = t_end - t_start;

    if (rank == 0) {
        printf("SOBEL done in %lf s\n", duration);
    }

#if !defined(SKIP_EXPORT)

    /* EXPORT Timer start */
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    /* Store file from array of pixels to GIF file */
    if (rank == 0 && store_pixels(output_filename, image)) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* EXPORT Timer stop */
    t_end = MPI_Wtime();

    duration = t_end - t_start;

    if (rank == 0) {
        printf("Export done in %lf s in file %s\n", duration, output_filename);
    }

#endif

    MPI_Finalize();
    return 0;
}
