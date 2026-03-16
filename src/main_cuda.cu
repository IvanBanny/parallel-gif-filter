/*
 * INF560
 *
 * Image Filtering Project
 */

#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cstddef>

#include "gif_io.h"
#include "helper_cuda.h"

typedef struct {
    uint32_t rgba;
} packed_pixel;

typedef struct {
    uint8_t grey;
} grey_pixel;

// KERNELS

__global__ void pack_pixels(pixel *in, packed_pixel *out, int pixel_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pixel_count) {
        const uint8_t r = (uint8_t)in[idx].r;
        const uint8_t g = (uint8_t)in[idx].g;
        const uint8_t b = (uint8_t)in[idx].b;

        const uint32_t rgb =
            ((uint32_t)r << 16) | ((uint32_t)g << 8) | ((uint32_t)b);

        out[idx].rgba = rgb;
    }
}

__global__ void unpack_pixels(packed_pixel *in, pixel *out, int pixel_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pixel_count) {
        uint32_t rgba = in[idx].rgba;
        out[idx].r = (rgba >> 16) & 0xFF;
        out[idx].g = (rgba >> 8) & 0xFF;
        out[idx].b = rgba & 0xFF;
    }
}

__global__ void grey_kernel(packed_pixel *color, grey_pixel *grey,
                            int pixel_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) {
        return;
    }

    uint32_t rgba = color[idx].rgba;
    uint32_t r = (rgba >> 16) & 0xFF;
    uint32_t g = (rgba >> 8) & 0xFF;
    uint32_t b = (rgba >> 0) & 0xFF;

    uint32_t val = (r + g + b) / 3;
    if (val > 255) {
        val = 255;
    }

    grey[idx].grey = (uint8_t)val;
}

__global__ void blur_kernel(grey_pixel *in, grey_pixel *out, int width,
                            int height, int size) {
    extern __shared__ uint8_t tile[];
    int shmem_w = blockDim.x + 2 * size;
    int shmem_h = blockDim.y + 2 * size;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load stencil into shared mem
    for (int ly = threadIdx.y; ly < shmem_h; ly += blockDim.y) {
        for (int lx = threadIdx.x; lx < shmem_w; lx += blockDim.x) {
            int gx = blockIdx.x * blockDim.x + lx - size;
            int gy = blockIdx.y * blockDim.y + ly - size;
            gx = max(0, min(gx, width - 1));
            gy = max(0, min(gy, height - 1));
            tile[ly * shmem_w + lx] = in[gy * width + gx].grey;
        }
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    // Copy everything first
    int idx = y * width + x;
    out[idx].grey = in[idx].grey;

    if (x < size || x >= width - size) {
        return;
    }

    int top_end = height / 10 - size;
    int bot_start = (int)(height - height / 10) + size;

    // Blur (top + bottom)
    if ((y >= size && y < top_end) || (y >= bot_start && y < height - size)) {
        int filter_area = (2 * size + 1) * (2 * size + 1);
        uint32_t total = 0;
        for (int dy = -size; dy <= size; ++dy) {
            for (int dx = -size; dx <= size; ++dx) {
                total += (uint32_t)tile[(threadIdx.y + size + dy) * shmem_w +
                                        threadIdx.x + size + dx];
            }
        }
        out[idx].grey = (uint8_t)(total / filter_area);
        return;
    }
}

__global__ void blur_check_kernel(grey_pixel *old_grey, grey_pixel *new_grey,
                                  int width, int height, int threshold,
                                  uint8_t *end) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return;
    }

    int idx = y * width + x;
    int diff = (int)new_grey[idx].grey - (int)old_grey[idx].grey;

    if (diff > threshold || -diff > threshold) {
        *end = 0;  // should be correct without atomic op since each thread
                   // writes the same value ?
    }
}

__global__ void sobel_kernel(grey_pixel *grey, packed_pixel *color,
                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        uint8_t g = grey[idx].grey;
        color[idx].rgba =
            ((uint32_t)g << 16) | ((uint32_t)g << 8) | (uint32_t)g;
        return;
    }

    int no = grey[(y - 1) * width + (x - 1)].grey;
    int n = grey[(y - 1) * width + (x)].grey;
    int ne = grey[(y - 1) * width + (x + 1)].grey;

    int o = grey[(y)*width + (x - 1)].grey;
    int e = grey[(y)*width + (x + 1)].grey;

    int so = grey[(y + 1) * width + (x - 1)].grey;
    int s = grey[(y + 1) * width + (x)].grey;
    int se = grey[(y + 1) * width + (x + 1)].grey;

    float dx = (float)(ne - no) + 2.0f * (e - o) + (float)(se - so);
    float dy = (float)(se - ne) + 2.0f * (s - n) + (float)(so - no);

    float val = sqrtf(dx * dx + dy * dy) / 4.0f;

    uint8_t g = (val > 50.0f) ? 255 : 0;

    uint32_t rgb = (g << 16) | (g << 8) | g;

    color[idx].rgba = rgb;
}

// CPU

void allocate_buffers(size_t max_pixel, pixel **h_buf,
                      pixel **d_unpacked_pixels, packed_pixel **color_image_gpu,
                      grey_pixel **grey_image_gpu1,
                      grey_pixel **grey_image_gpu2, uint8_t **h_end_flag,
                      uint8_t **d_end_flag) {
    for (int i = 0; i < 2; ++i) {
        checkCudaErrors(
            cudaMallocHost((void **)&h_buf[i], max_pixel * sizeof(pixel)));
        checkCudaErrors(cudaMalloc((void **)&d_unpacked_pixels[i],
                                   max_pixel * sizeof(pixel)));
        checkCudaErrors(cudaMalloc((void **)&color_image_gpu[i],
                                   max_pixel * sizeof(packed_pixel)));
        checkCudaErrors(cudaMalloc((void **)&grey_image_gpu1[i],
                                   max_pixel * sizeof(grey_pixel)));
        checkCudaErrors(cudaMalloc((void **)&grey_image_gpu2[i],
                                   max_pixel * sizeof(grey_pixel)));
    }
    checkCudaErrors(cudaMallocHost((void **)h_end_flag, sizeof(uint8_t)));
    checkCudaErrors(cudaMalloc((void **)d_end_flag, sizeof(uint8_t)));
}

void send_data_to_gpu(animated_gif *image, int index, pixel *h_buf,
                      pixel *d_unpacked_pixels, packed_pixel *color_image_gpu,
                      cudaStream_t stream) {
    const size_t pixel_count =
        (size_t)image->width[index] * image->height[index];

    // Copy pixels to pinned buffer on host and async copy to device
    memcpy(h_buf, image->p[index], pixel_count * sizeof(pixel));
    checkCudaErrors(cudaMemcpyAsync(d_unpacked_pixels, h_buf,
                                    pixel_count * sizeof(pixel),
                                    cudaMemcpyHostToDevice, stream));

    // Pack pixels on device
    int block = 256;
    int grid = (int)((pixel_count + block - 1) / block);
    pack_pixels<<<grid, block, 0, stream>>>(d_unpacked_pixels, color_image_gpu,
                                            (int)pixel_count);
    checkCudaErrors(cudaGetLastError());
}

void apply_grey_filter(animated_gif *image, int index,
                       packed_pixel *color_image_gpu,
                       grey_pixel *grey_image_gpu, cudaStream_t stream) {
    size_t pixel_count = (size_t)image->width[index] * image->height[index];

    int block = 256;
    int grid = (int)((pixel_count + block - 1) / block);

    grey_kernel<<<grid, block, 0, stream>>>(color_image_gpu, grey_image_gpu,
                                            (int)pixel_count);
    checkCudaErrors(cudaGetLastError());
}

void apply_blur_filter(animated_gif *image, int index, int size, int threshold,
                       uint8_t *h_end_flag, uint8_t *d_end_flag,
                       grey_pixel *grey_in, grey_pixel *grey_temp,
                       cudaStream_t stream) {
    int width = image->width[index];
    int height = image->height[index];

    grey_pixel *in = grey_in;
    grey_pixel *out = grey_temp;

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    int shmem_size =
        (block.x + 2 * size) * (block.y + 2 * size) * sizeof(uint8_t);

    do {
        checkCudaErrors(
            cudaMemsetAsync(d_end_flag, 1, sizeof(uint8_t), stream));

        blur_kernel<<<grid, block, shmem_size, stream>>>(in, out, width, height,
                                                         size);
        checkCudaErrors(cudaGetLastError());

        blur_check_kernel<<<grid, block, 0, stream>>>(in, out, width, height,
                                                      threshold, d_end_flag);
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpyAsync(h_end_flag, d_end_flag, sizeof(uint8_t),
                                        cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));

        if (*h_end_flag == 1) {
            break;
        }

        grey_pixel *tmp = in;
        in = out;
        out = tmp;
    } while (threshold > 0);

    if (out != grey_in) {
        size_t pixel_count = (size_t)width * height;
        checkCudaErrors(cudaMemcpyAsync(grey_in, out,
                                        pixel_count * sizeof(grey_pixel),
                                        cudaMemcpyDeviceToDevice, stream));
    }
}

void apply_sobel_filter(animated_gif *image, int index, grey_pixel *d_grey,
                        packed_pixel *d_color, cudaStream_t stream) {
    int width = image->width[index];
    int height = image->height[index];

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    sobel_kernel<<<grid, block, 0, stream>>>(d_grey, d_color, width, height);
    checkCudaErrors(cudaGetLastError());
}

void fetch_data_from_gpu(animated_gif *image, int index, pixel *h_buf,
                         pixel *d_unpacked_pixels,
                         packed_pixel *color_image_gpu, cudaStream_t stream) {
    size_t pixel_count =
        (size_t)image->width[index] * image->height[index];

    // Unpack pixels on device
    int block = 256;
    int grid = (int)((pixel_count + block - 1) / block);
    unpack_pixels<<<grid, block, 0, stream>>>(color_image_gpu,
                                              d_unpacked_pixels,
                                              (int)pixel_count);
    checkCudaErrors(cudaGetLastError());

    // Async copy pixel array to host
    checkCudaErrors(cudaMemcpyAsync(h_buf, d_unpacked_pixels,
                                    pixel_count * sizeof(pixel),
                                    cudaMemcpyDeviceToHost, stream));
}

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
    checkCudaErrors(cudaFree(0));

    // FILTER Timer start
    gettimeofday(&t1, NULL);

    size_t max_pixels = 0;
    for (int i = 0; i < image->n_images; ++i) {
        size_t pixels = (size_t)image->width[i] * (size_t)image->height[i];
        if (pixels > max_pixels) {
            max_pixels = pixels;
        }
    }

    pixel *h_buf[2] = {NULL, NULL};  // pinned host buffer for async copy
    pixel *d_unpacked_pixels[2] = {NULL, NULL};
    packed_pixel *d_color[2] = {NULL, NULL};
    grey_pixel *d_grey1[2] = {NULL, NULL};
    grey_pixel *d_grey2[2] = {NULL, NULL};
    uint8_t *h_end_flag = NULL;
    uint8_t *d_end_flag = NULL;
    allocate_buffers(max_pixels, h_buf, d_unpacked_pixels, d_color, d_grey1,
                     d_grey2, &h_end_flag, &d_end_flag);

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    for (int i = 0; i < image->n_images; ++i) {
        int s = i % 2;

        // If this slot had a previous frame - download it
        if (i >= 2) {
            // Wait for this slot's previous work to finish
            checkCudaErrors(cudaStreamSynchronize(streams[s]));

            memcpy(image->p[i - 2], h_buf[s],
                   (size_t)image->width[i - 2] * image->height[i - 2] *
                       sizeof(pixel));
        }

        // Async upload
        send_data_to_gpu(image, i, h_buf[s], d_unpacked_pixels[s], d_color[s],
                         streams[s]);

        // Run kernels on streams[s]
        apply_grey_filter(image, i, d_color[s], d_grey1[s], streams[s]);
        apply_blur_filter(image, i, 5, 20, h_end_flag, d_end_flag, d_grey1[s],
                          d_grey2[s], streams[s]);
        apply_sobel_filter(image, i, d_grey1[s], d_color[s], streams[s]);

        // Async download back to pinned buffer
        fetch_data_from_gpu(image, i, h_buf[s], d_unpacked_pixels[s],
                            d_color[s], streams[s]);
    }

    // Drain
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy remaining frames from pinned buffers
    for (int i = image->n_images > 1 ? image->n_images - 2 : 0;
         i < image->n_images; ++i) {
        int s = i % 2;
        memcpy(image->p[i], h_buf[s],
               (size_t)image->width[i] * image->height[i] * sizeof(pixel));
    }

    // Free buffers
    for (int i = 0; i < 2; ++i) {
        cudaFreeHost(h_buf[i]);
        cudaFree(d_unpacked_pixels[i]);
        cudaFree(d_color[i]);
        cudaFree(d_grey1[i]);
        cudaFree(d_grey2[i]);
    }
    cudaFreeHost(h_end_flag);
    cudaFree(d_end_flag);

    // Free streams
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    // FILTER Timer stop
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("SOBEL done in %lf s\n", duration);

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

    return 0;
}
