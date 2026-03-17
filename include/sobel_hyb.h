#pragma once
#ifndef SOBEL_HYB_H
#define SOBEL_HYB_H

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gif_io.h"

static void die_mpi_hyb(const char* message) 
{
    fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static void* malloc_bytes_hyb(size_t nbytes, const char* what) 
{
    void* ptr = malloc(nbytes);
    if (ptr == NULL && nbytes != 0) 
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Unable to allocate memory for %s", what);
        die_mpi_hyb(buffer);
    }
    return ptr;
}

static animated_gif* load_input_on_root_hyb(int rank, const char* input_filename, int* n_images_out)
{
    animated_gif* image = NULL;

    if (rank == 0)
    {
        image = load_pixels((char*)input_filename);
        
        if (image == NULL) { die_mpi_hyb("Unable to load input GIF"); }

        *n_images_out = image->n_images;
    }

    MPI_Bcast(n_images_out, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return image;
}

static void broadcast_dimensions_hyb(animated_gif* image, int rank, int n_images, int** width_out, int** height_out)
{
    int* width = (int*)malloc_bytes_hyb((size_t)n_images * sizeof(int), "width array");
    int* height = (int*)malloc_bytes_hyb((size_t)n_images * sizeof(int), "height array");

    if (rank == 0)
    {
        memcpy(width, image->width, (size_t)n_images * sizeof(int));
        memcpy(height, image->height, (size_t)n_images * sizeof(int));
    }

    MPI_Bcast(width, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(height, n_images, MPI_INT, 0, MPI_COMM_WORLD);

    *width_out = width;
    *height_out = height;
}

static void apply_gray_filter_hyb(pixel *p, int width, int height) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < width * height; ++i) {
        int avg = (p[i].r + p[i].g + p[i].b) / 3;
        if (avg < 0) avg = 0;
        if (avg > 255) avg = 255;

        p[i].r = avg;
        p[i].g = avg;
        p[i].b = avg;
    }
}

static void apply_blur_filter_hyb(pixel *p, int width, int height, int size,
                       int threshold) {
    int end = 0, n_iter = 0;

    /* Allocate array of new_img pixels */
    pixel *new_img = (pixel*)malloc(width * height * sizeof(pixel));
    if (new_img == NULL) {
        fprintf(stderr, "Unable to allocate memory for new_img blur pixels\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Perform at least one blur iteration */
    do {
        end = 1;
        ++n_iter;

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < height - 1; ++i) {
                for (int j = 0; j < width - 1; ++j) {
                    new_img[CONV(i, j, width)].r = p[CONV(i, j, width)].r;
                    new_img[CONV(i, j, width)].g = p[CONV(i, j, width)].g;
                    new_img[CONV(i, j, width)].b = p[CONV(i, j, width)].b;
                }
            }

            /* Apply blur on top part of image (10%) */
            #pragma omp for schedule(static) nowait
            for (int i = size; i < height / 10 - size; ++i) {
                for (int j = size; j < width - size; ++j) {
                    int stencil_i, stencil_j;
                    int t_r = 0, t_g = 0, t_b = 0;

                    for (stencil_i = -size; stencil_i <= size; ++stencil_i) {
                        for (stencil_j = -size; stencil_j <= size;
                            ++stencil_j) {
                            t_r +=
                                p[CONV(i + stencil_i, j + stencil_j, width)].r;
                            t_g +=
                                p[CONV(i + stencil_i, j + stencil_j, width)].g;
                            t_b +=
                                p[CONV(i + stencil_i, j + stencil_j, width)].b;
                        }
                    }

                    new_img[CONV(i, j, width)].r =
                        t_r / ((2 * size + 1) * (2 * size + 1));
                    new_img[CONV(i, j, width)].g =
                        t_g / ((2 * size + 1) * (2 * size + 1));
                    new_img[CONV(i, j, width)].b =
                        t_b / ((2 * size + 1) * (2 * size + 1));
                }
            }

            /* Copy the middle part of the image */
            #pragma omp for schedule(static) nowait
            for (int i = height / 10 - size; i < height - height / 10 + size;
                ++i) {
                for (int j = size; j < width - size; ++j) {
                    new_img[CONV(i, j, width)].r = p[CONV(i, j, width)].r;
                    new_img[CONV(i, j, width)].g = p[CONV(i, j, width)].g;
                    new_img[CONV(i, j, width)].b = p[CONV(i, j, width)].b;
                }
            }

            /* Apply blur on the bottom part of the image (10%) */
            #pragma omp for schedule(static) nowait
            for (int i = height - height / 10 + size; i < height - size; ++i) {
                for (int j = size; j < width - size; ++j) {
                    int stencil_i, stencil_j;
                    int t_r = 0, t_g = 0, t_b = 0;

                    for (stencil_i = -size; stencil_i <= size; ++stencil_i) {
                        for (stencil_j = -size; stencil_j <= size;
                            ++stencil_j) {
                            t_r +=
                                p[CONV(i + stencil_i, j + stencil_j, width)].r;
                            t_g +=
                                p[CONV(i + stencil_i, j + stencil_j, width)].g;
                            t_b +=
                                p[CONV(i + stencil_i, j + stencil_j, width)].b;
                        }
                    }

                    new_img[CONV(i, j, width)].r =
                        t_r / ((2 * size + 1) * (2 * size + 1));
                    new_img[CONV(i, j, width)].g =
                        t_g / ((2 * size + 1) * (2 * size + 1));
                    new_img[CONV(i, j, width)].b =
                        t_b / ((2 * size + 1) * (2 * size + 1));
                }
            }
        }

        #pragma omp parallel for schedule(static) reduction(&& : end)
        for (int i = 1; i < height - 1; ++i) {
            for (int j = 1; j < width - 1; ++j) {
                float diff_r, diff_g, diff_b;
                diff_r =
                    (new_img[CONV(i, j, width)].r - p[CONV(i, j, width)].r);
                diff_g =
                    (new_img[CONV(i, j, width)].g - p[CONV(i, j, width)].g);
                diff_b =
                    (new_img[CONV(i, j, width)].b - p[CONV(i, j, width)].b);

                if (diff_r > threshold || -diff_r > threshold ||
                    diff_g > threshold || -diff_g > threshold ||
                    diff_b > threshold || -diff_b > threshold) {
                    end = 0;
                }

                p[CONV(i, j, width)].r = new_img[CONV(i, j, width)].r;
                p[CONV(i, j, width)].g = new_img[CONV(i, j, width)].g;
                p[CONV(i, j, width)].b = new_img[CONV(i, j, width)].b;
            }
        }

    } while (threshold > 0 && !end);

#if SOBELF_DEBUG
    printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

    free(new_img);
}

static void apply_sobel_filter_hyb(pixel *p, int width, int height) {
    pixel *sobel = (pixel*)malloc(width * height * sizeof(pixel));
    if (sobel == NULL) {
        fprintf(stderr, "Unable to allocate memory for new sobel pixels\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

#pragma omp parallel for schedule(static)
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            int pixel_blue_nw, pixel_blue_n, pixel_blue_ne;
            int pixel_blue_sw, pixel_blue_s, pixel_blue_se;
            int pixel_blue_w, pixel_blue_e;

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

#pragma omp parallel for schedule(static)
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
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

static int cmp_desc(const void *a, const void *b) {
    const frame_info *fa = (const frame_info *)a;
    const frame_info *fb = (const frame_info *)b;
    return (fb->count > fa->count) - (fb->count < fa->count);
}

void sobel_hyb(animated_gif* image, int n_images, int rank, int n_ranks)
{
    int* width;
    int* height;
    broadcast_dimensions_hyb(image, rank, n_images, &width, &height);

    /* Count pixels and argsort descending */
    frame_info *fi = (frame_info*)malloc(n_images * sizeof(frame_info));
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
    int *load = (int*)calloc(n_ranks, sizeof(int));
    int *owner = (int*)malloc(n_images * sizeof(int));
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
    pixel **local_pixels = (pixel**) malloc(n_images * sizeof(pixel *));
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
            local_pixels[i] = (pixel*)malloc(frame_size);
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
        apply_gray_filter_hyb(local_pixels[i], width[i], height[i]);
        apply_blur_filter_hyb(local_pixels[i], width[i], height[i], 5, 20);
        apply_sobel_filter_hyb(local_pixels[i], width[i], height[i]);
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

    for (int i = 0; i < n_images; ++i) {
        if (local_pixels[i]) {
            free(local_pixels[i]);
        }
    }
    free(local_pixels);
    free(owner);

    free(width);
    free(height);
}

#endif