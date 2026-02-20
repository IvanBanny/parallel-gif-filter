/*
 * INF560
 *
 * Image Filtering Project
 */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "gif_io.h"

void apply_gray_filter(animated_gif *image) {
    pixel **p = image->p;
    for (int i = 0; i < image->n_images; ++i) {
        const int n_px = image->width[i] * image->height[i];
#pragma omp parallel for schedule(static)
        for (int j = 0; j < n_px; ++j) {
            int avg = (p[i][j].r + p[i][j].g + p[i][j].b) / 3;
            if (avg < 0) avg = 0;
            if (avg > 255) avg = 255;
            p[i][j].r = avg;
            p[i][j].g = avg;
            p[i][j].b = avg;
        }
    }
}

void apply_gray_line(animated_gif *image) {
    pixel **p = image->p;
    for (int i = 0; i < image->n_images; ++i) {
        for (int j = 0; j < 10; ++j) {
#pragma omp parallel for schedule(static)
            for (int k = image->width[i] / 2; k < image->width[i]; ++k) {
                p[i][CONV(j, k, image->width[i])].r = 0;
                p[i][CONV(j, k, image->width[i])].g = 0;
                p[i][CONV(j, k, image->width[i])].b = 0;
            }
        }
    }
}

void apply_blur_filter(animated_gif *image, int size, int threshold) {
    int width, height, end = 0, n_iter = 0;

    pixel **p = image->p;
    pixel *new;

    /* Process all images */
    for (int i = 0; i < image->n_images; ++i) {
        n_iter = 0;
        width = image->width[i];
        height = image->height[i];

        /* Allocate array of new pixels */
        new = (pixel *)malloc(width * height * sizeof(pixel));

        /* Perform at least one blur iteration */
        do {
            end = 1;
            ++n_iter;

#pragma omp parallel for schedule(static)
            for (int j = 0; j < height - 1; ++j) {
                for (int k = 0; k < width - 1; ++k) {
                    new[CONV(j, k, width)].r = p[i][CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[i][CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[i][CONV(j, k, width)].b;
                }
            }

            /* Apply blur on top part of image (10%) */
#pragma omp parallel for schedule(static)
            for (int j = size; j < height / 10 - size; ++j) {
                for (int k = size; k < width - size; ++k) {
                    int stencil_j, stencil_k;
                    int t_r = 0;
                    int t_g = 0;
                    int t_b = 0;

                    for (stencil_j = -size; stencil_j <= size; ++stencil_j) {
                        for (stencil_k = -size; stencil_k <= size;
                             ++stencil_k) {
                            t_r +=
                                p[i][CONV(j + stencil_j, k + stencil_k, width)]
                                    .r;
                            t_g +=
                                p[i][CONV(j + stencil_j, k + stencil_k, width)]
                                    .g;
                            t_b +=
                                p[i][CONV(j + stencil_j, k + stencil_k, width)]
                                    .b;
                        }
                    }

                    new[CONV(j, k, width)].r =
                        t_r / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].g =
                        t_g / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].b =
                        t_b / ((2 * size + 1) * (2 * size + 1));
                }
            }

            /* Copy the middle part of the image */
#pragma omp parallel for schedule(static)
            for (int j = height / 10 - size; j < height - height / 10 + size;
                 ++j) {
                for (int k = size; k < width - size; ++k) {
                    new[CONV(j, k, width)].r = p[i][CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[i][CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[i][CONV(j, k, width)].b;
                }
            }

            /* Apply blur on the bottom part of the image (10%) */
#pragma omp parallel for schedule(static)
            for (int j = height - height / 10 + size; j < height - size; ++j) {
                for (int k = size; k < width - size; ++k) {
                    int stencil_j, stencil_k;
                    int t_r = 0;
                    int t_g = 0;
                    int t_b = 0;

                    for (stencil_j = -size; stencil_j <= size; ++stencil_j) {
                        for (stencil_k = -size; stencil_k <= size;
                             ++stencil_k) {
                            t_r +=
                                p[i][CONV(j + stencil_j, k + stencil_k, width)]
                                    .r;
                            t_g +=
                                p[i][CONV(j + stencil_j, k + stencil_k, width)]
                                    .g;
                            t_b +=
                                p[i][CONV(j + stencil_j, k + stencil_k, width)]
                                    .b;
                        }
                    }

                    new[CONV(j, k, width)].r =
                        t_r / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].g =
                        t_g / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].b =
                        t_b / ((2 * size + 1) * (2 * size + 1));
                }
            }

#pragma omp parallel for schedule(static) reduction(&&:end)
            for (int j = 1; j < height - 1; ++j) {
                for (int k = 1; k < width - 1; ++k) {
                    float diff_r;
                    float diff_g;
                    float diff_b;

                    diff_r =
                        (new[CONV(j, k, width)].r - p[i][CONV(j, k, width)].r);
                    diff_g =
                        (new[CONV(j, k, width)].g - p[i][CONV(j, k, width)].g);
                    diff_b =
                        (new[CONV(j, k, width)].b - p[i][CONV(j, k, width)].b);

                    if (diff_r > threshold || -diff_r > threshold ||
                        diff_g > threshold || -diff_g > threshold ||
                        diff_b > threshold || -diff_b > threshold) {
                        end = 0;
                    }

                    p[i][CONV(j, k, width)].r = new[CONV(j, k, width)].r;
                    p[i][CONV(j, k, width)].g = new[CONV(j, k, width)].g;
                    p[i][CONV(j, k, width)].b = new[CONV(j, k, width)].b;
                }
            }

        } while (threshold > 0 && !end);

#if SOBELF_DEBUG
        printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

        free(new);
    }
}

void apply_sobel_filter(animated_gif *image) {
    int width, height;

    pixel **p = image->p;

    for (int i = 0; i < image->n_images; ++i) {
        width = image->width[i];
        height = image->height[i];

        pixel *sobel;

        sobel = (pixel *)malloc(width * height * sizeof(pixel));

#pragma omp parallel for schedule(static)
        for (int j = 1; j < height - 1; ++j) {
            for (int k = 1; k < width - 1; ++k) {
                int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                int pixel_blue_so, pixel_blue_s, pixel_blue_se;
                int pixel_blue_o, pixel_blue, pixel_blue_e;

                float deltaX_blue;
                float deltaY_blue;
                float val_blue;

                pixel_blue_no = p[i][CONV(j - 1, k - 1, width)].b;
                pixel_blue_n = p[i][CONV(j - 1, k, width)].b;
                pixel_blue_ne = p[i][CONV(j - 1, k + 1, width)].b;
                pixel_blue_so = p[i][CONV(j + 1, k - 1, width)].b;
                pixel_blue_s = p[i][CONV(j + 1, k, width)].b;
                pixel_blue_se = p[i][CONV(j + 1, k + 1, width)].b;
                pixel_blue_o = p[i][CONV(j, k - 1, width)].b;
                pixel_blue = p[i][CONV(j, k, width)].b;
                pixel_blue_e = p[i][CONV(j, k + 1, width)].b;

                deltaX_blue = -pixel_blue_no + pixel_blue_ne -
                              2 * pixel_blue_o + 2 * pixel_blue_e -
                              pixel_blue_so + pixel_blue_se;

                deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so -
                              pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;

                val_blue = sqrt(deltaX_blue * deltaX_blue +
                                deltaY_blue * deltaY_blue) /
                           4;

                if (val_blue > 50) {
                    sobel[CONV(j, k, width)].r = 255;
                    sobel[CONV(j, k, width)].g = 255;
                    sobel[CONV(j, k, width)].b = 255;
                } else {
                    sobel[CONV(j, k, width)].r = 0;
                    sobel[CONV(j, k, width)].g = 0;
                    sobel[CONV(j, k, width)].b = 0;
                }
            }
        }

#pragma omp parallel for schedule(static)
        for (int j = 1; j < height - 1; ++j) {
            for (int k = 1; k < width - 1; ++k) {
                p[i][CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
                p[i][CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
                p[i][CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
            }
        }

        free(sobel);
    }
}

/*
 * Main entry point
 */
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

    /* Convert the pixels into grayscale */
    apply_gray_filter(image);

    /* Apply blur filter with convergence value */
    apply_blur_filter(image, 5, 20);

    /* Apply sobel filter on pixels */
    apply_sobel_filter(image);

    /* FILTER Timer stop */
    t_end = omp_get_wtime();

    duration = t_end - t_start;

    printf("SOBEL done in %lf s\n", duration);

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

    return 0;
}
