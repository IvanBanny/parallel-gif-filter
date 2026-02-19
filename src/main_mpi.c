/*
 * INF560
 *
 * Image Filtering Project
 * MPI Implementation
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gif_io.h"

void apply_gray_filter(pixel *p, int width, int height) {
    int i;
    for (i = 0; i < width * height; ++i) {
        int avg = (p[i].r + p[i].g + p[i].b) / 3;
        if (avg < 0) avg = 0;
        if (avg > 255) avg = 255;

        p[i].r = avg;
        p[i].g = avg;
        p[i].b = avg;
    }
}

void apply_gray_line(pixel *p, int width) {
    int i, j;
    for (i = 0; i < 10; ++i) {
        for (j = width / 2; j < width; ++j) {
            p[CONV(i, j, width)].r = 0;
            p[CONV(i, j, width)].g = 0;
            p[CONV(i, j, width)].b = 0;
        }
    }
}

void apply_blur_filter(pixel *p, int width, int height, int size,
                       int threshold) {
    int i, j;
    int end = 0;
    int n_iter = 0;

    /* Allocate array of new pixels */
    pixel *new = malloc(width * height * sizeof(pixel));
    if (new == NULL) {
        fprintf(stderr, "Unable to allocate memory for new blur pixels\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Perform at least one blur iteration */
    do {
        end = 1;
        ++n_iter;

        for (i = 0; i < height - 1; ++i) {
            for (j = 0; j < width - 1; ++j) {
                new[CONV(i, j, width)].r = p[CONV(i, j, width)].r;
                new[CONV(i, j, width)].g = p[CONV(i, j, width)].g;
                new[CONV(i, j, width)].b = p[CONV(i, j, width)].b;
            }
        }

        /* Apply blur on top part of image (10%) */
        for (i = size; i < height / 10 - size; ++i) {
            for (j = size; j < width - size; ++j) {
                int stencil_i, stencil_j;
                int t_r = 0, t_g = 0, t_b = 0;

                for (stencil_i = -size; stencil_i <= size; ++stencil_i) {
                    for (stencil_j = -size; stencil_j <= size; ++stencil_j) {
                        t_r += p[CONV(i + stencil_i, j + stencil_j, width)].r;
                        t_g += p[CONV(i + stencil_i, j + stencil_j, width)].g;
                        t_b += p[CONV(i + stencil_i, j + stencil_j, width)].b;
                    }
                }

                new[CONV(i, j, width)].r =
                    t_r / ((2 * size + 1) * (2 * size + 1));
                new[CONV(i, j, width)].g =
                    t_g / ((2 * size + 1) * (2 * size + 1));
                new[CONV(i, j, width)].b =
                    t_b / ((2 * size + 1) * (2 * size + 1));
            }
        }

        /* Copy the middle part of the image */
        for (i = height / 10 - size; i < height - height / 10 + size; ++i) {
            for (j = size; j < width - size; ++j) {
                new[CONV(i, j, width)].r = p[CONV(i, j, width)].r;
                new[CONV(i, j, width)].g = p[CONV(i, j, width)].g;
                new[CONV(i, j, width)].b = p[CONV(i, j, width)].b;
            }
        }

        /* Apply blur on the bottom part of the image (10%) */
        for (i = height - height / 10 + size; i < height - size; ++i) {
            for (j = size; j < width - size; ++j) {
                int stencil_i, stencil_j;
                int t_r = 0, t_g = 0, t_b = 0;

                for (stencil_i = -size; stencil_i <= size; ++stencil_i) {
                    for (stencil_j = -size; stencil_j <= size; ++stencil_j) {
                        t_r += p[CONV(i + stencil_i, j + stencil_j, width)].r;
                        t_g += p[CONV(i + stencil_i, j + stencil_j, width)].g;
                        t_b += p[CONV(i + stencil_i, j + stencil_j, width)].b;
                    }
                }

                new[CONV(i, j, width)].r =
                    t_r / ((2 * size + 1) * (2 * size + 1));
                new[CONV(i, j, width)].g =
                    t_g / ((2 * size + 1) * (2 * size + 1));
                new[CONV(i, j, width)].b =
                    t_b / ((2 * size + 1) * (2 * size + 1));
            }
        }

        for (i = 1; i < height - 1; ++i) {
            for (j = 1; j < width - 1; ++j) {
                float diff_r, diff_g, diff_b;
                diff_r = (new[CONV(i, j, width)].r - p[CONV(i, j, width)].r);
                diff_g = (new[CONV(i, j, width)].g - p[CONV(i, j, width)].g);
                diff_b = (new[CONV(i, j, width)].b - p[CONV(i, j, width)].b);

                if (diff_r > threshold || -diff_r > threshold ||
                    diff_g > threshold || -diff_g > threshold ||
                    diff_b > threshold || -diff_b > threshold) {
                    end = 0;
                }

                p[CONV(i, j, width)].r = new[CONV(i, j, width)].r;
                p[CONV(i, j, width)].g = new[CONV(i, j, width)].g;
                p[CONV(i, j, width)].b = new[CONV(i, j, width)].b;
            }
        }

    } while (threshold > 0 && !end);

#if SOBELF_DEBUG
    printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

    free(new);
}

void apply_sobel_filter(pixel *p, int width, int height) {
    int i, j;

    pixel *sobel = malloc(width * height * sizeof(pixel));
    if (sobel == NULL) {
        fprintf(stderr, "Unable to allocate memory for new sobel pixels\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (i = 1; i < height - 1; ++i) {
        for (j = 1; j < width - 1; ++j) {
            int pixel_blue_nw, pixel_blue_n, pixel_blue_ne;
            int pixel_blue_sw, pixel_blue_s, pixel_blue_se;
            int pixel_blue_w, pixel_blue, pixel_blue_e;

            float deltaX_blue;
            float deltaY_blue;
            float val_blue;

            pixel_blue_nw = p[CONV(i - 1, j - 1, width)].b;
            pixel_blue_n = p[CONV(i - 1, j, width)].b;
            pixel_blue_ne = p[CONV(i - 1, j + 1, width)].b;
            pixel_blue_sw = p[CONV(i + 1, j - 1, width)].b;
            pixel_blue_s = p[CONV(i + 1, j, width)].b;
            pixel_blue_se = p[CONV(i + 1, j + 1, width)].b;
            pixel_blue_w = p[CONV(i, j - 1, width)].b;
            pixel_blue = p[CONV(i, j, width)].b;
            pixel_blue_e = p[CONV(i, j + 1, width)].b;

            deltaX_blue = -pixel_blue_nw + pixel_blue_ne - 2 * pixel_blue_w +
                          2 * pixel_blue_e - pixel_blue_sw + pixel_blue_se;

            deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_sw -
                          pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_nw;

            val_blue =
                sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;

            if (val_blue > 50) {
                sobel[CONV(i, j, width)].r = 255;
                sobel[CONV(i, j, width)].g = 255;
                sobel[CONV(i, j, width)].b = 255;
            } else {
                sobel[CONV(i, j, width)].r = 0;
                sobel[CONV(i, j, width)].g = 0;
                sobel[CONV(i, j, width)].b = 0;
            }
        }
    }

    for (i = 1; i < height - 1; ++i) {
        for (j = 1; j < width - 1; ++j) {
            p[CONV(i, j, width)].r = sobel[CONV(i, j, width)].r;
            p[CONV(i, j, width)].g = sobel[CONV(i, j, width)].g;
            p[CONV(i, j, width)].b = sobel[CONV(i, j, width)].b;
        }
    }

    free(sobel);
}

typedef struct {
    int idx;
    int count;
} frame_info;

int cmp_desc(const void *a, const void *b) {
    const frame_info *fa = (const frame_info *)a;
    const frame_info *fb = (const frame_info *)b;
    return (fb->count > fa->count) - (fb->count < fa->count);
}

int main(int argc, char **argv) {
    int rank, n_ranks;
    char *input_filename;
    char *output_filename;
    animated_gif *image;
    double t_start, t_end;
    double duration;

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
    if (rank == 0) {
        /* Load file and store the pixels in array */
        image = load_pixels(input_filename);
        if (image == NULL) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n_images = image->n_images;
    }
    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *width = malloc(n_images * sizeof(int));
    int *height = malloc(n_images * sizeof(int));
    if (width == NULL || height == NULL) {
        fprintf(stderr, "Unable to allocate memory for width/height\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0) {
        memcpy(width, image->width, n_images * sizeof(int));
        memcpy(height, image->height, n_images * sizeof(int));
    }
    MPI_Bcast(width, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(height, n_images, MPI_INT, 0, MPI_COMM_WORLD);

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

    /* Count pixels and argsort descending */
    frame_info *fi = malloc(n_images * sizeof(frame_info));
    if (fi == NULL) {
        fprintf(stderr, "Unable to allocate frame info array\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < n_images; ++i) {
        fi[i].idx = i;
        fi[i].count = width[i] * height[i];
    }
    qsort(fi, n_images, sizeof(frame_info), cmp_desc);

    /* Greedy assignment */
    int *load = calloc(n_ranks, sizeof(int));
    int *owner = malloc(n_images * sizeof(int));
    if (load == NULL || owner == NULL) {
        fprintf(stderr, "Unable to allocate load/owner array\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int s = 0; s < n_images; ++s) {
        int frame = fi[s].idx;
        int min_r = 0;
        for (int r = 1; r < n_ranks; ++r) {
            if (load[r] < load[min_r]) {
                min_r = r;
            }
        }
        owner[frame] = min_r;
        load[min_r] += fi[s].count;
    }
    free(fi);
    free(load);

    /* Scatter frames to ranks */
    pixel **local_pixels = malloc(n_images * sizeof(pixel *));
    if (local_pixels == NULL) {
        fprintf(stderr, "Unable to allocate memory for local frames\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < n_images; ++i) {
        int frame_size = (int)(width[i] * height[i] * sizeof(pixel));

        if (rank == 0 && owner[i] != 0) {
            MPI_Send(image->p[i], frame_size, MPI_BYTE, owner[i], i,
                     MPI_COMM_WORLD);
            local_pixels[i] = NULL;
        } else if (rank == owner[i]) {
            local_pixels[i] = malloc(frame_size);
            if (local_pixels[i] == NULL) {
                fprintf(stderr,
                        "Unable to allocate memory for a local frame\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (rank == 0) {
                memcpy(local_pixels[i], image->p[i], frame_size);
            } else {
                MPI_Recv(local_pixels[i], frame_size, MPI_BYTE, 0, i,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            local_pixels[i] = NULL;
        }
    }

    /* Apply filters */
    for (int i = 0; i < n_images; ++i) {
        if (rank != owner[i]) {
            continue;
        }
        apply_gray_filter(local_pixels[i], width[i], height[i]);
        apply_blur_filter(local_pixels[i], width[i], height[i], 5, 20);
        apply_sobel_filter(local_pixels[i], width[i], height[i]);
    }

    /* Gather frames to 0 */
    for (int i = 0; i < n_images; ++i) {
        int frame_size = (int)(width[i] * height[i] * sizeof(pixel));

        if (rank == 0 && owner[i] != 0) {
            MPI_Recv(image->p[i], frame_size, MPI_BYTE, owner[i], i,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == owner[i] && rank != 0) {
            MPI_Send(local_pixels[i], frame_size, MPI_BYTE, 0, i,
                     MPI_COMM_WORLD);
        } else if (rank == 0) {
            memcpy(image->p[i], local_pixels[i], frame_size);
        }
    }

    /* FILTER Timer stop */
    t_end = MPI_Wtime();

    duration = t_end - t_start;

    if (rank == 0) {
        printf("SOBEL done in %lf s\n", duration);
    }

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

    for (int i = 0; i < n_images; ++i) {
        if (local_pixels[i]) {
            free(local_pixels[i]);
        }
    }
    free(local_pixels);
    free(owner);
    free(width);
    free(height);

    MPI_Finalize();
    return 0;
}
