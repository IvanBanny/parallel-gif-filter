#pragma once
#ifndef SOBEL_MPI_H
#define SOBEL_MPI_H

#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gif_io.h"

#define BLUR_SIZE 5
#define BLUR_THRESHOLD 20

typedef struct 
{
    int idx;
    size_t pixels;
} 
frame_info_mpi;

typedef struct 
{
    int global_idx;
    int width;
    int height;
    pixel* data;
    int borrowed_from_image;
} 
local_frame;

typedef struct 
{
    int frame_idx;
    int width;
    int height;
    int start_row;
    int local_rows; // number of local rows
    int top_halo;
    int bottom_halo;
    pixel* data;
}
local_stripe;


// == UTILS == //

static void die_mpi(const char* message) 
{
    fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static void* malloc_bytes(size_t nbytes, const char* what) 
{
    void* ptr = malloc(nbytes);
    if (ptr == NULL && nbytes != 0) 
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Unable to allocate memory for %s", what);
        die_mpi(buffer);
    }
    return ptr;
}

static void* calloc_bytes(size_t count, size_t elem_size, const char* what) 
{
    void* ptr = calloc(count, elem_size);
    if (ptr == NULL && count != 0 && elem_size != 0) 
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Unable to allocate memory for %s", what);
        die_mpi(buffer);
    }
    return ptr;
}

static size_t frame_pixels(int width, int height) 
{
    if (width < 0 || height < 0) 
    {
        die_mpi("Negative frame dimension encountered");
    }
    return (size_t)width * (size_t)height;
}

static size_t frame_nbytes(int width, int height) 
{
    return frame_pixels(width, height) * sizeof(pixel);
}


// ==_LOAD == //

static animated_gif* load_input_on_root(int rank, const char* input_filename, int* n_images_out)
{
    animated_gif* image = NULL;

    if (rank == 0)
    {
        image = load_pixels((char*)input_filename);
        
        if (image == NULL) { die_mpi("Unable to load input GIF"); }

        *n_images_out = image->n_images;
    }

    MPI_Bcast(n_images_out, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return image;
}

static void broadcast_dimensions(animated_gif* image, int rank, int n_images, int** width_out, int** height_out)
{
    int* width = (int*)malloc_bytes((size_t)n_images * sizeof(int), "width array");
    int* height = (int*)malloc_bytes((size_t)n_images * sizeof(int), "height array");

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

// == FRAME FILTERS == //

static void apply_gray_filter_mpi(pixel* p, int width, int height) 
{
    size_t n = frame_pixels(width, height);
    for (size_t i = 0; i < n; i++) 
    {
        int avg = (p[i].r + p[i].g + p[i].b) / 3;
        if (avg < 0) avg = 0;
        if (avg > 255) avg = 255;

        p[i].r = avg;
        p[i].g = avg;
        p[i].b = avg;
    }
}

static void apply_blur_filter_mpi(pixel* p, int width, int height, int size, int threshold) 
{
    const int kernel_width = 2 * size + 1;
    const int kernel_area = kernel_width * kernel_width;
    const int top_end = height / 10 - size;
    const int bottom_begin = height - height / 10 + size;

    if (width <= 0 || height <= 0) { return; }
    if (size < 0) { die_mpi("Negative blur size encountered"); }
    if (width < kernel_width || height < kernel_width) { return; }

    pixel* new_pixels = (pixel*)malloc_bytes(frame_nbytes(width, height), "new blur pixels");

    int end = 0;
    int n_iter = 0;
    do 
    {
        end = 1;
        n_iter++;

        memcpy(new_pixels, p, frame_nbytes(width, height));

        /* Apply blur on top part of image (10%) */
        for (int i = size; i < top_end; i++) 
        {
            for (int j = size; j < width - size; j++) 
            {
                int t_r = 0, t_g = 0, t_b = 0;

                for (int stencil_i = -size; stencil_i <= size; stencil_i++) 
                {
                    for (int stencil_j = -size; stencil_j <= size; stencil_j++) 
                    {
                        t_r += p[CONV(i + stencil_i, j + stencil_j, width)].r;
                        t_g += p[CONV(i + stencil_i, j + stencil_j, width)].g;
                        t_b += p[CONV(i + stencil_i, j + stencil_j, width)].b;
                    }
                }

                new_pixels[CONV(i, j, width)].r = t_r / kernel_area;
                new_pixels[CONV(i, j, width)].g = t_g / kernel_area;
                new_pixels[CONV(i, j, width)].b = t_b / kernel_area;
            }
        }

        /* Apply blur on the bottom part of the image (10%) */
        for (int i = bottom_begin; i < height - size; i++) 
        {
            for (int j = size; j < width - size; j++) 
            {
                int t_r = 0, t_g = 0, t_b = 0;

                for (int stencil_i = -size; stencil_i <= size; stencil_i++) 
                {
                    for (int stencil_j = -size; stencil_j <= size; stencil_j++) 
                    {
                        t_r += p[CONV(i + stencil_i, j + stencil_j, width)].r;
                        t_g += p[CONV(i + stencil_i, j + stencil_j, width)].g;
                        t_b += p[CONV(i + stencil_i, j + stencil_j, width)].b;
                    }
                }

                new_pixels[CONV(i, j, width)].r = t_r / kernel_area;
                new_pixels[CONV(i, j, width)].g = t_g / kernel_area;
                new_pixels[CONV(i, j, width)].b = t_b / kernel_area;
            }
        }

        for (int i = 1; i < height - 1; i++) 
        {
            for (int j = 1; j < width - 1; j++) 
            {
                float diff_r = (float)new_pixels[CONV(i, j, width)].r - (float)p[CONV(i, j, width)].r;
                float diff_g = (float)new_pixels[CONV(i, j, width)].g - (float)p[CONV(i, j, width)].g;
                float diff_b = (float)new_pixels[CONV(i, j, width)].b - (float)p[CONV(i, j, width)].b;

                if (diff_r > threshold || -diff_r > threshold || diff_g > threshold || -diff_g > threshold || diff_b > threshold || -diff_b > threshold) 
                {
                    end = 0;
                }

                p[CONV(i, j, width)].r = new_pixels[CONV(i, j, width)].r;
                p[CONV(i, j, width)].g = new_pixels[CONV(i, j, width)].g;
                p[CONV(i, j, width)].b = new_pixels[CONV(i, j, width)].b;
            }
        }
    } 
    while (threshold > 0 && !end);

#if SOBELF_DEBUG
    printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

    free(new_pixels);
}

static void apply_sobel_filter_mpi(pixel* p, int width, int height) 
{
    if (width < 3 || height < 3) { return; }

    pixel* sobel = (pixel*)malloc_bytes(frame_nbytes(width, height), "new sobel pixels");
    memcpy(sobel, p, frame_nbytes(width, height));

    for (int i = 1; i < height - 1; i++) 
    {
        for (int j = 1; j < width - 1; j++) 
        {
            int n  = p[CONV(i - 1, j, width)].b;
            int ne = p[CONV(i - 1, j + 1, width)].b;
            int e  = p[CONV(i, j + 1, width)].b;
            int se = p[CONV(i + 1, j + 1, width)].b;
            int s  = p[CONV(i + 1, j, width)].b;
            int sw = p[CONV(i + 1, j - 1, width)].b;
            int w  = p[CONV(i, j - 1, width)].b;
            int nw = p[CONV(i - 1, j - 1, width)].b;

            float delta_x = (ne - nw) + 2 * (e - w) + (se - sw);
            float delta_y = (se - ne) + 2 * (s - n) + (sw - nw);
            float val = sqrtf(delta_x * delta_x + delta_y * delta_y) / 4.0f;

            int gray_value = (val > 50.0f) ? 255 : 0;

            sobel[CONV(i, j, width)].r = gray_value;
            sobel[CONV(i, j, width)].g = gray_value;
            sobel[CONV(i, j, width)].b = gray_value;
        }
    }

    for (int i = 1; i < height - 1; i++) 
    {
        for (int j = 1; j < width - 1; j++) 
        {
            p[CONV(i, j, width)].r = sobel[CONV(i, j, width)].r;
            p[CONV(i, j, width)].g = sobel[CONV(i, j, width)].g;
            p[CONV(i, j, width)].b = sobel[CONV(i, j, width)].b;
        }
    }

    free(sobel);
}

// == FRAME MODE == //

static int cmp_frame_info_desc(const void* a, const void* b) 
{
    const frame_info_mpi* fa = (const frame_info_mpi*)a;
    const frame_info_mpi* fb = (const frame_info_mpi*)b;
    if (fb->pixels > fa->pixels) { return 1; }
    if (fb->pixels < fa->pixels) { return -1; }
    return 0;
}

static int* compute_frame_owners(const int* width, const int* height, int n_images, int n_ranks)
{
    frame_info_mpi* info = (frame_info_mpi*)malloc_bytes((size_t)n_images * sizeof(frame_info_mpi), "frame info array");
    size_t* load = (size_t*)calloc_bytes((size_t)n_ranks, sizeof(size_t), "rank loads");
    int* owner = (int*)malloc_bytes((size_t)n_images * sizeof(int), "owner array");

    for (int i = 0; i < n_images; i++)
    {
        info[i].idx = i;
        info[i].pixels = frame_pixels(width[i], height[i]);
    }

    qsort(info, (size_t)n_images, sizeof(frame_info_mpi), cmp_frame_info_desc);

    for (int s = 0; s < n_images; s++)
    {
        int frame = info[s].idx;

        int min_rank = 0;
        for (int r = 1; r < n_ranks; r++)
        {
            if (load[r] < load[min_rank])
            {
                min_rank = r;
            }
        }

        owner[frame] = min_rank;
        load[min_rank] += info[s].pixels;
    }

    free(load);
    free(info);
    return owner;
}

static int count_local_frames(const int* owner, int n_images, int rank)
{
    int count = 0;
    for (int i = 0; i < n_images; i++)
    {
        if (owner[i] == rank)
        {
            count++;
        }
    }
    return count;
}

static local_frame* distribute_frames(animated_gif* image, const int* width, const int* height, const int* owner, int n_images, int rank, int my_n_frames) 
{
    local_frame* my_frames = (local_frame*)calloc_bytes((size_t)my_n_frames, sizeof(local_frame), "local frame descriptors");

    // send data
    if (rank == 0)
    {
        for (int i = 0; i < n_images; i++)
        {
            if (owner[i] == 0) { continue; }

            size_t nbytes = frame_nbytes(width[i], height[i]);
            MPI_Send(image->p[i], (int)nbytes, MPI_BYTE, owner[i], i, MPI_COMM_WORLD);
        }
    }

    // receive data
    int k = 0;
    for (int i = 0; i < n_images; i++)
    {
        if (owner[i] != rank) { continue; }

        my_frames[k].global_idx = i;
        my_frames[k].width = width[i];
        my_frames[k].height = height[i];

        if (rank == 0)
        {
            my_frames[k].data = image->p[i];
            my_frames[k].borrowed_from_image = 1;
        }
        else
        {
            size_t nbytes = frame_nbytes(width[i], height[i]);
            my_frames[k].data = (pixel*)malloc_bytes(nbytes, "local frame data");
            my_frames[k].borrowed_from_image = 0;
            MPI_Recv(my_frames[k].data, (int)nbytes, MPI_BYTE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        k++;
    }

    return my_frames;
}

static void process_local_frames(local_frame* my_frames, int my_n_frames)
{
    for (int k = 0; k < my_n_frames; k++)
    {
        apply_gray_filter_mpi(my_frames[k].data, my_frames[k].width, my_frames[k].height);
        apply_blur_filter_mpi(my_frames[k].data, my_frames[k].width, my_frames[k].height, 5, 20);
        apply_sobel_filter_mpi(my_frames[k].data, my_frames[k].width, my_frames[k].height);
    }
}

static void gather_frames(animated_gif* image, const int* width, const int* height,
                          const int* owner, int n_images, int rank,
                          const local_frame* my_frames, int my_n_frames) 
{
    if (rank == 0)
    {
        for (int i = 0; i < n_images; i++)
        {
            if (owner[i] == 0) { continue; }

            size_t nbytes = frame_nbytes(width[i], height[i]);
            MPI_Recv(image->p[i], (int)nbytes, MPI_BYTE, owner[i], i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        return;
    }

    for (int k = 0; k < my_n_frames; k++)
    {
        size_t nbytes = frame_nbytes(my_frames[k].width, my_frames[k].height);
        MPI_Send(my_frames[k].data, (int)nbytes, MPI_BYTE, 0, my_frames[k].global_idx, MPI_COMM_WORLD);
    }
}

static void free_local_frames(local_frame* my_frames, int my_n_frames)
{
    for (int k = 0; k < my_n_frames; k++)
    {
        if (my_frames[k].data != NULL && !my_frames[k].borrowed_from_image)
        {
            free(my_frames[k].data);
        }
    }

    free(my_frames);
}

static void run_frame_mode(animated_gif* image, const int* width, const int* height, int n_images, int rank, int n_ranks) 
{
    // Distributing
    int* owner = compute_frame_owners(width, height, n_images, n_ranks);
    int my_n_frames = count_local_frames(owner, n_images, rank);
    local_frame* my_frames = distribute_frames(image, width, height, owner, n_images, rank, my_n_frames);

    // Filtering
    process_local_frames(my_frames, my_n_frames);

    // Gathering
    gather_frames(image, width, height, owner, n_images, rank, my_frames, my_n_frames);

    // free memory
    free_local_frames(my_frames, my_n_frames);
    free(owner);
}


// == STRIPE UTILS == //

static void compute_block_partition(int total, int part_rank, int n_parts, int* start_out, int* count_out)
{
    int base = total / n_parts;
    int extra = total % n_parts;
    int count = base + (part_rank < extra ? 1 : 0);
    int start = part_rank * base + (part_rank < extra ? part_rank : extra);

    *start_out = start;
    *count_out = count;
}

static pixel* stripe_owned_ptr(const local_stripe* stripe)
{
    return stripe->data + (size_t)stripe->top_halo * (size_t)stripe->width;
}

static int stripe_total_rows(const local_stripe* stripe)
{
    return stripe->top_halo + stripe->local_rows + stripe->bottom_halo;
}

static void exchange_halos(local_stripe* stripe, int halo_rows, int frame_rank, int frame_size, MPI_Comm frame_comm) 
{
    pixel* owned = stripe_owned_ptr(stripe);
    int width = stripe->width;
    int up = (frame_rank > 0) ? frame_rank - 1 : MPI_PROC_NULL;
    int down = (frame_rank + 1 < frame_size) ? frame_rank + 1 : MPI_PROC_NULL;
    int row_nbytes = width * (int)sizeof(pixel);

    if (halo_rows <= 0 || frame_size <= 1 || stripe->local_rows == 0) { return; }
    if (stripe->local_rows < halo_rows) 
    {
        die_mpi("Row-wise decomposition is thinner than blur halo size");
    }

    MPI_Sendrecv(owned,
                 halo_rows * row_nbytes, MPI_BYTE, up, 0,
                 stripe->data,
                 halo_rows * row_nbytes, MPI_BYTE, up, 1,
                 frame_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(owned + (size_t)(stripe->local_rows - halo_rows) * (size_t)width,
                 halo_rows * row_nbytes, MPI_BYTE, down, 1,
                 owned + (size_t)stripe->local_rows * (size_t)width,
                 halo_rows * row_nbytes, MPI_BYTE, down, 0,
                 frame_comm, MPI_STATUS_IGNORE);
}


// == STRIPE FILTERS == //

static void apply_gray_filter_stripe(local_stripe* stripe) 
{
    pixel* owned = stripe_owned_ptr(stripe);
    size_t n = (size_t)stripe->local_rows * (size_t)stripe->width;

    for (size_t i = 0; i < n; i++)
    {
        int avg = (owned[i].r + owned[i].g + owned[i].b) / 3;
        if (avg < 0) { avg = 0; }
        if (avg > 255) { avg = 255; }

        owned[i].r = avg;
        owned[i].g = avg;
        owned[i].b = avg;
    }
}

static void apply_blur_filter_stripe(local_stripe* stripe, int size, int threshold, int frame_rank, int frame_size, MPI_Comm frame_comm)
{
    int width = stripe->width;
    int height = stripe->height;
    int local_rows = stripe->local_rows;
    int total_rows = stripe_total_rows(stripe);
    pixel* owned = stripe_owned_ptr(stripe);

    int kernel_width = 2 * size + 1;
    int kernel_area = kernel_width * kernel_width;
    int top_end = height / 10 - size;
    int bottom_begin = height - height / 10 + size;

    if (width <= 0 || height <= 0 || local_rows == 0) { return; }
    if (size < 0) { die_mpi("Negative blur size encountered"); }
    if (width < kernel_width || height < kernel_width) { return; }

    pixel* new_pixels = (pixel*)malloc_bytes((size_t)total_rows * (size_t)width * sizeof(pixel), "stripe blur pixels");

    int global_end = 0;
    do
    {
        int local_end = 1;

        exchange_halos(stripe, size, frame_rank, frame_size, frame_comm);
        memcpy(new_pixels, stripe->data, (size_t)total_rows * (size_t)width * sizeof(pixel));

        
        for (int local_i = 0; local_i < local_rows; local_i++) 
        {
            // get location in stripe
            int data_i = stripe->top_halo + local_i;

            // get location in frame
            int global_i = stripe->start_row + local_i;
            int in_top_band = (global_i >= size && global_i < top_end);
            int in_bottom_band = (global_i >= bottom_begin && global_i < height - size);

            // skip middle part
            if (!in_top_band && !in_bottom_band) { continue; }
            
            /* Apply blur on top and bottom part of the image (10%)*/
            for (int j = size; j < width - size; j++) 
            {
                int t_r = 0;
                int t_g = 0;
                int t_b = 0;

                for (int stencil_i = -size; stencil_i <= size; stencil_i++) 
                {
                    for (int stencil_j = -size; stencil_j <= size; stencil_j++) 
                    {
                        pixel px = stripe->data[CONV(data_i + stencil_i, j + stencil_j, width)];
                        t_r += px.r;
                        t_g += px.g;
                        t_b += px.b;
                    }
                }

                new_pixels[CONV(data_i, j, width)].r = t_r / kernel_area;
                new_pixels[CONV(data_i, j, width)].g = t_g / kernel_area;
                new_pixels[CONV(data_i, j, width)].b = t_b / kernel_area;
            }
        }

        // check local end condition
        for (int local_i = 0; local_i < local_rows; local_i++) 
        {
            int data_i = stripe->top_halo + local_i;
            int global_i = stripe->start_row + local_i;

            if (global_i <= 0 || global_i >= height - 1) { continue; }

            for (int j = 1; j < width - 1; j++) 
            {
                float diff_r = (float) new_pixels[CONV(data_i, j, width)].r - (float) stripe->data[CONV(data_i, j, width)].r;
                float diff_g = (float) new_pixels[CONV(data_i, j, width)].g - (float) stripe->data[CONV(data_i, j, width)].g;
                float diff_b = (float) new_pixels[CONV(data_i, j, width)].b - (float) stripe->data[CONV(data_i, j, width)].b;

                if (diff_r > threshold || -diff_r > threshold || diff_g > threshold || -diff_g > threshold || diff_b > threshold || -diff_b > threshold) 
                {
                    local_end = 0;
                }
            }
        }

        // copy the new values into the stripe
        memcpy(owned, new_pixels + (size_t)stripe->top_halo * (size_t)width,
               (size_t)local_rows * (size_t)width * sizeof(pixel));

        // check global end condition
        MPI_Allreduce(&local_end, &global_end, 1, MPI_INT, MPI_LAND, frame_comm);
    }
    while(threshold > 0 && !global_end);

    free(new_pixels);
}

static void apply_sobel_filter_stripe(local_stripe *stripe, int frame_rank, int frame_size, MPI_Comm frame_comm) 
{
    int width = stripe->width;
    int height = stripe->height;
    int local_rows = stripe->local_rows;
    int total_rows = stripe_total_rows(stripe);

    if (width < 3 || height < 3 || local_rows == 0) { return; }

    exchange_halos(stripe, 1, frame_rank, frame_size, frame_comm); // halos are not up to date at the end of blur filter

    pixel* sobel = (pixel*)malloc_bytes((size_t)total_rows * (size_t)width * sizeof(pixel), "stripe sobel pixels");
    memcpy(sobel, stripe->data, (size_t)total_rows * (size_t)width * sizeof(pixel));

    for (int local_i = 0; local_i < local_rows; local_i++) 
    {
        int global_i = stripe->start_row + local_i;
        int data_i = stripe->top_halo + local_i;

        if (global_i <= 0 || global_i >= height - 1) { continue; }

        for (int j = 1; j < width - 1; j++) 
        {
            int n  = stripe->data[CONV(data_i - 1, j, width)].b;
            int ne = stripe->data[CONV(data_i - 1, j + 1, width)].b;
            int e  = stripe->data[CONV(data_i, j + 1, width)].b;
            int se = stripe->data[CONV(data_i + 1, j + 1, width)].b;
            int s  = stripe->data[CONV(data_i + 1, j, width)].b;
            int sw = stripe->data[CONV(data_i + 1, j - 1, width)].b;
            int w  = stripe->data[CONV(data_i, j - 1, width)].b;
            int nw = stripe->data[CONV(data_i - 1, j - 1, width)].b;

            float delta_x = (ne - nw) + 2 * (e - w) + (se - sw);
            float delta_y = (se - ne) + 2 * (s - n) + (sw - nw);
            float val = sqrtf(delta_x * delta_x + delta_y * delta_y) / 4.0f;

            int gray_value = (val > 50.0f) ? 255 : 0;

            sobel[CONV(data_i, j, width)].r = gray_value;
            sobel[CONV(data_i, j, width)].g = gray_value;
            sobel[CONV(data_i, j, width)].b = gray_value;
        }
    }

    memcpy(stripe_owned_ptr(stripe), sobel + (size_t)stripe->top_halo * (size_t)width, (size_t)local_rows * (size_t)width * sizeof(pixel));

    free(sobel);
}


// == STRIPE MODE == //

static void build_frame_groups(int n_images, int n_ranks, int** frame_group_size_out, int** rank_to_frame_out, int** rank_in_group_out) 
{
    int* frame_group_size = (int*)malloc_bytes((size_t)n_images * sizeof(int), "frame group size");
    int* rank_to_frame = (int*)malloc_bytes((size_t)n_ranks * sizeof(int), "rank to frame");
    int* rank_in_group = (int*)malloc_bytes((size_t)n_ranks * sizeof(int), "rank in group");

    int base = n_ranks / n_images;
    int extra = n_ranks % n_images;

    int cursor = 0;
    for (int f = 0; f < n_images; f++) 
    {
        int group_size = base + (f < extra ? 1 : 0);

        frame_group_size[f] = group_size;
        for (int j = 0; j < group_size; j++) 
        {
            rank_to_frame[cursor + j] = f;
            rank_in_group[cursor + j] = j;
        }

        cursor += group_size;
    }

    *frame_group_size_out = frame_group_size;
    *rank_to_frame_out = rank_to_frame;
    *rank_in_group_out = rank_in_group;
}

static local_stripe distribute_stripes(animated_gif* image, const int* width, const int* height,
                                         const int* rank_to_frame, const int* rank_in_group,
                                         const int* frame_group_size, int rank, int n_ranks) 
{
    local_stripe stripe;
    memset(&stripe, 0, sizeof(stripe));

    int my_frame = rank_to_frame[rank];
    int my_group_rank = rank_in_group[rank];
    int my_group_size = frame_group_size[my_frame];

    stripe.frame_idx = my_frame;
    stripe.width = width[my_frame];
    stripe.height = height[my_frame];

    int start_row = 0;
    int local_rows = 0;
    compute_block_partition(stripe.height, my_group_rank, my_group_size, &start_row, &local_rows);

    stripe.start_row = start_row;
    stripe.local_rows = local_rows;

    stripe.top_halo = (my_group_rank > 0) ? BLUR_SIZE : 0;
    stripe.bottom_halo = (my_group_rank + 1 < my_group_size) ? BLUR_SIZE : 0;

    stripe.data = (pixel*)malloc_bytes((size_t)(stripe.top_halo + stripe.local_rows + stripe.bottom_halo) * (size_t)stripe.width * sizeof(pixel), "row stripe");

    pixel* owned_ptr = stripe_owned_ptr(&stripe);

    if (rank == 0) 
    {
        for (int r = 0; r < n_ranks; r++) 
        {
            int frame = rank_to_frame[r];
            int group_rank = rank_in_group[r];
            int group_size = frame_group_size[frame];
            int r_start = 0;
            int r_rows = 0;
            pixel* src = NULL;
            size_t nbytes = 0;

            compute_block_partition(height[frame], group_rank, group_size, &r_start, &r_rows);
            src = image->p[frame] + (size_t)r_start * (size_t)width[frame];
            nbytes = (size_t)r_rows * (size_t)width[frame] * sizeof(pixel);

            if (r == 0) 
            {
                memcpy(owned_ptr, src, nbytes);
            } 
            else 
            {
                MPI_Send(src, (int)nbytes, MPI_BYTE, r, frame, MPI_COMM_WORLD);
            }
        }
    } 
    else 
    {
        size_t recv_nbytes = (size_t)stripe.local_rows * (size_t)stripe.width * sizeof(pixel);
        MPI_Recv(owned_ptr, (int)recv_nbytes, MPI_BYTE, 0, my_frame, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (stripe.local_rows > 0 && my_group_size > 1 && stripe.local_rows < BLUR_SIZE) 
    {
        die_mpi("Too many ranks per frame for blur halo size");
    }

    return stripe;
}

static void process_stripes(local_stripe *stripe, int frame_rank, int frame_size, MPI_Comm frame_comm) 
{
    apply_gray_filter_stripe(stripe);
    apply_blur_filter_stripe(stripe, BLUR_SIZE, BLUR_THRESHOLD, frame_rank, frame_size, frame_comm);
    apply_sobel_filter_stripe(stripe, frame_rank, frame_size, frame_comm);
}

static void gather_stripes(animated_gif* image, const local_stripe *stripe, const int* rank_to_frame, const int* rank_in_group, 
                             const int* frame_group_size, int rank, int n_ranks, int n_images) 
{
    pixel* owned = stripe_owned_ptr(stripe);
    size_t my_nbytes = (size_t)stripe->local_rows * (size_t)stripe->width * sizeof(pixel);

    if (rank == 0) 
    {
        memcpy(image->p[stripe->frame_idx] + (size_t)stripe->start_row * (size_t)stripe->width, owned, my_nbytes);

        for (int r = 1; r < n_ranks; r++) 
        {
            int frame = rank_to_frame[r];
            int group_rank = rank_in_group[r];
            int group_size = frame_group_size[frame];
            
            int start_row = 0;
            int local_rows = 0;
            compute_block_partition(image->height[frame], group_rank, group_size, &start_row, &local_rows);
            size_t recv_nbytes = (size_t)local_rows * (size_t)image->width[frame] * sizeof(pixel);
            pixel* dst = image->p[frame] + (size_t)start_row * (size_t)image->width[frame];
            int gather_tag = frame + n_images;

            MPI_Recv(dst, (int)recv_nbytes, MPI_BYTE, r, gather_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else 
    {
        int gather_tag = stripe->frame_idx + n_images;
        MPI_Send(owned, (int)my_nbytes, MPI_BYTE, 0, gather_tag, MPI_COMM_WORLD);
    }
}

static void free_local_stripe(local_stripe *stripe) 
{
    free(stripe->data);
}

static void run_grouped_row_mode(animated_gif* image, const int* width, const int* height, int n_images, int rank, int n_ranks) 
{
    int* frame_group_size = NULL;
    int* rank_to_frame = NULL;
    int* rank_in_group = NULL;

    int my_frame = 0;
    MPI_Comm frame_comm = MPI_COMM_NULL;
    int frame_rank = 0;
    int frame_size = 0;
    local_stripe stripe;

    build_frame_groups(n_images, n_ranks, &frame_group_size, &rank_to_frame, &rank_in_group);

    my_frame = rank_to_frame[rank];
    MPI_Comm_split(MPI_COMM_WORLD, my_frame, rank, &frame_comm);
    MPI_Comm_rank(frame_comm, &frame_rank);
    MPI_Comm_size(frame_comm, &frame_size);

    // Distributing
    stripe = distribute_stripes(image, width, height, rank_to_frame, rank_in_group, frame_group_size, rank, n_ranks);

    // Filtering
    process_stripes(&stripe, frame_rank, frame_size, frame_comm);

    // Gathering
    gather_stripes(image, &stripe, rank_to_frame, rank_in_group, frame_group_size, rank, n_ranks, n_images);

    // free memory
    free_local_stripe(&stripe);
    MPI_Comm_free(&frame_comm);
    free(frame_group_size);
    free(rank_to_frame);
    free(rank_in_group);
}

void sobel_mpi(animated_gif* image, int n_images, int rank, int n_ranks)
{
    int* width;
    int* height;
    broadcast_dimensions(image, rank, n_images, &width, &height);

    // Choose distribution method (should be based on the number of frames and the size of each one)
    if (n_images * 2 >= n_ranks) 
    {
        if(rank == 0) { printf("one rank per image\n"); }

        run_frame_mode(image, width, height, n_images, rank, n_ranks);
    }
    else 
    {
        if(rank == 0) { printf("multiple rank per image\n"); }

        run_grouped_row_mode(image, width, height, n_images, rank, n_ranks);
    }

    free(width);
    free(height);
}

#endif