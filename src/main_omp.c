/*
 * INF560
 *
 * Image Filtering Project
 */

#include "sobel_omp.h"

int main(int argc, char **argv) {
    char *input_filename, *output_filename;
    animated_gif *image;
    double t_start, t_end, duration;

    /* Check command-line arguments */
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.gif output.gif \n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];

    /* IMPORT Timer start */
    t_start = omp_get_wtime();

    /* Load file and store the pixels in array */
    image = load_pixels(input_filename);
    if (image == NULL) {
        return 1;
    }

    /* IMPORT Timer stop */
    t_end = omp_get_wtime();

    duration = t_end - t_start;

    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
           input_filename, image->n_images, duration);

    /* FILTER Timer start */
    t_start = omp_get_wtime();

    sobel_omp(image);

    /* FILTER Timer stop */
    t_end = omp_get_wtime();

    duration = t_end - t_start;

    printf("SOBEL done in %lf s\n", duration);

#if !defined(SKIP_EXPORT)

    /* EXPORT Timer start */
    t_start = omp_get_wtime();

    /* Store file from array of pixels to GIF file */
    if (store_pixels(output_filename, image)) {
        return 1;
    }

    /* EXPORT Timer stop */
    t_end = omp_get_wtime();

    duration = t_end - t_start;

    printf("Export done in %lf s in file %s\n", duration, output_filename);

#endif

    return 0;
}
