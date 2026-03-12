/*
 * INF560
 *
 * Image Filtering Project
 * MPI Implementation
 */
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gif_io.h"

typedef struct 
{
    int idx;
    size_t pixels;
}
frame_info;

typedef struct
{
    int global_idx;
    int width;
    int height;
    pixel* data;
    int borrowed_from_image; /* 1 only for root-owned frames */
}
local_frame;

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

// == FILTERS == //

static void apply_gray_filter(pixel* p, int width, int height) 
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

static void apply_blur_filter(pixel* p, int width, int height, int size, int threshold) 
{
    int end = 0;
    int n_iter = 0;

    const int kernel_width = 2 * size + 1;
    const int kernel_area = kernel_width * kernel_width;
    const int top_end = height / 10 - size;
    const int bottom_begin = height - height / 10 + size;

    if (width <= 0 || height <= 0) { return; }
    if (size < 0) { die_mpi("Negative blur size encountered"); }
    if (width < kernel_width || height < kernel_width) { return; }

    pixel* new_pixels = malloc_bytes(frame_nbytes(width, height), "new blur pixels");

    do {
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
                float diff_r = (float)new_pixels[CONV(i, j, width)].r -
                               (float)p[CONV(i, j, width)].r;
                float diff_g = (float)new_pixels[CONV(i, j, width)].g -
                               (float)p[CONV(i, j, width)].g;
                float diff_b = (float)new_pixels[CONV(i, j, width)].b -
                               (float)p[CONV(i, j, width)].b;

                if (diff_r > threshold || -diff_r > threshold ||
                    diff_g > threshold || -diff_g > threshold ||
                    diff_b > threshold || -diff_b > threshold) 
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

static void apply_sobel_filter(pixel* p, int width, int height) 
{
    if (width < 3 || height < 3) { return; }

    pixel* sobel = malloc_bytes(frame_nbytes(width, height), "new sobel pixels");
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
    int* width = malloc_bytes((size_t)n_images * sizeof(int), "width array");
    int* height = malloc_bytes((size_t)n_images * sizeof(int), "height array");

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

// == DISTRIBUTIAN == //

static int cmp_frame_info_desc(const void* a, const void* b) 
{
    const frame_info* fa = (const frame_info*)a;
    const frame_info* fb = (const frame_info*)b;
    if (fb->pixels > fa->pixels) { return 1; }
    if (fb->pixels < fa->pixels) { return -1; }
    return 0;
}

static int* compute_frame_owners(const int* width, const int* height, int n_images, int n_ranks)
{
    frame_info* info = malloc_bytes((size_t)n_images * sizeof(frame_info), "frame info array");
    size_t* load = calloc_bytes((size_t)n_ranks, sizeof(size_t), "rank loads");
    int* owner = malloc_bytes((size_t)n_images * sizeof(int), "owner array");

    for (int i = 0; i < n_images; i++)
    {
        info[i].idx = i;
        info[i].pixels = frame_pixels(width[i], height[i]);
    }

    qsort(info, (size_t)n_images, sizeof(frame_info), cmp_frame_info_desc);

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

static void distribute_frames(animated_gif* image, const int* width, const int* height, 
                              const int* owner, int n_images, int rank, int my_n_frames, local_frame** my_frames_out) 
{
    local_frame* my_frames = calloc_bytes((size_t)my_n_frames, sizeof(local_frame), "local frame descriptors");

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
            my_frames[k].data = malloc_bytes(nbytes, "local frame data");
            my_frames[k].borrowed_from_image = 0;
            MPI_Recv(my_frames[k].data, (int)nbytes, MPI_BYTE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        k++;
    }

    *my_frames_out = my_frames;
}

// == Filter == //

static void process_local_frames(local_frame* my_frames, int my_n_frames)
{
    for (int k = 0; k < my_n_frames; k++)
    {
        apply_gray_filter(my_frames[k].data, my_frames[k].width, my_frames[k].height);
        apply_blur_filter(my_frames[k].data, my_frames[k].width, my_frames[k].height, 5, 20);
        apply_sobel_filter(my_frames[k].data, my_frames[k].width, my_frames[k].height);
    }
}

// == GATHER == //

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

// ==_MAIN == //

int main(int argc, char** argv)
{
    int rank = 0;
    int n_ranks = 0;
    int n_images = 0;
    int* width = NULL;
    int* height = NULL;
    int* owner = NULL;
    int my_n_frames = 0;
    local_frame* my_frames = NULL;
    animated_gif* image = NULL;
    double t_start = 0.0, t_end = 0.0, duration = 0.0;
    const char* input_filename = NULL;
    const char* output_filename = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (argc < 3) {
        if (rank == 0) {
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
    broadcast_dimensions(image, rank, n_images, &width, &height);
    
    t_end = MPI_Wtime();
    duration = t_end - t_start;
    if (rank == 0)
    {
        printf("GIF loaded from file %s with %d image(s) in %lf s\n", input_filename, n_images, duration);
    }

    // Actual work
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    // Distributing
    owner = compute_frame_owners(width, height, n_images, n_ranks);
    my_n_frames = count_local_frames(owner, n_images, rank);
    distribute_frames(image, width, height, owner, n_images, rank, my_n_frames, &my_frames);

    // Filtering
    process_local_frames(my_frames, my_n_frames);

    // Gathering
    gather_frames(image, width, height, owner, n_images, rank,
                  my_frames, my_n_frames);

    t_end = MPI_Wtime();
    duration = t_end - t_start;
    if (rank == 0) {
        printf("SOBEL done in %lf s\n", duration);
    }

    // Output gif
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    if (rank == 0 && store_pixels((char*)output_filename, image)) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    t_end = MPI_Wtime();
    duration = t_end - t_start;
    if (rank == 0) {
        printf("Export done in %lf s in file %s\n", duration, output_filename);
    }

    // free memory
    free_local_frames(my_frames, my_n_frames);
    free(owner);
    free(width);
    free(height);

    MPI_Finalize();
    return 0;
}
