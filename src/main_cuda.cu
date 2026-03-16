/*
 * INF560
 *
 * Image Filtering Project
 */

#include "sobel_cuda.h"

int main(int argc, char **argv) {
    char *input_filename, *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;

    // Check command-line arguments
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.gif output.gif \n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];

    // IMPORT Timer start
    gettimeofday(&t1, NULL);

    // Load file and store the pixels in array
    image = load_pixels(input_filename);
    if (image == NULL) {
        return 1;
    }

    // IMPORT Timer stop
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
           input_filename, image->n_images, duration);

    // Warmup for fair comparison
    cuda_warmup();

    // FILTER Timer start
    gettimeofday(&t1, NULL);

    sobel_cuda(image);

    // FILTER Timer stop
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("SOBEL done in %lf s\n", duration);

#if !defined(SKIP_EXPORT)

    // EXPORT Timer start
    gettimeofday(&t1, NULL);

    // Store file from array of pixels to GIF file
    if (store_pixels(output_filename, image)) {
        return 1;
    }

    // EXPORT Timer stop
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("Export done in %lf s in file %s\n", duration, output_filename);

#endif

    return 0;
}
