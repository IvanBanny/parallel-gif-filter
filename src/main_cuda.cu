/*
 * INF560
 *
 * Image Filtering Project
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>

#include "gif_io.h"
#include "helper_cuda.h"


typedef struct
{
    uint32_t rgba;
} packed_pixel;

typedef struct
{
    uint8_t gray;
} gray_pixel;


// KERNELS

__global__ void gray_kernel(packed_pixel *color_gpu, gray_pixel *gray_gpu1, int pixel_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= pixel_count)
    {
        return;
    }

    uint32_t rgba = color_gpu[idx].rgba;
    uint32_t r = (rgba >> 16) & 0xFF;
    uint32_t g = (rgba >> 8) & 0xFF;
    uint32_t b = (rgba >> 0) & 0xFF;

    uint32_t gray = (r + g + b) / 3;
    if(gray > 255) { gray = 255; }

    gray_gpu1[idx].gray = (uint8_t)gray;
}


__global__ void blur_kernel(gray_pixel *in, gray_pixel *out, int width, int height, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) { return; }

    // copy everything first
    int idx = y * width + x;
    out[idx].gray = in[idx].gray;

    if(x < size || x >= width - size)
    {
        return;
    }

    int top_end = height / 10 - size;
    int bot_start = (int)(height * 0.9) + size;

    int filter_area = (2 * size + 1) * (2 * size + 1);

    // top part
    if(y >= size && y < top_end)
    {
        uint32_t total = 0;
        for(int dy = -size; dy <= size; dy++)
        {
            for(int dx = -size; dx <= size; dx++)
            {
                total += (uint32_t)in[(y + dy) * width + x + dx].gray;
            }
        }
        out[idx].gray = (uint8_t)(total / filter_area);
        return;
    }

    // middule part
    if(y >= top_end && y < bot_start)
    {
        return;
    }

    // bottom part
    if(y >= bot_start && y < height - size)
    {
        uint32_t total = 0;
        for(int dy = -size; dy <= size; dy++)
        {
            for(int dx = -size; dx <= size; dx++)
            {
                total += (uint32_t)in[(y + dy) * width + x + dx].gray;
            }
        }
        out[idx].gray = (uint8_t)(total / filter_area);
        return;
    }
}

__global__ void blur_check_kernel(gray_pixel* old_gray, gray_pixel* new_gray, int width, int height, int threshold, uint8_t* end)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) { return; }

    int idx = y * width + x;
    int diff = (int)new_gray[idx].gray - (int)old_gray[idx].gray;

    if(diff > threshold || -diff > threshold)
    {
        *end = 0; // should be correct without atomic op since each thread writes the same value ?
    }
}


__global__ void sobel_kernel(gray_pixel* gray_gpu1, packed_pixel* color_gpu, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) { return; }

    int idx = y * width + x;

    if(x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)
    {
        uint8_t g = gray_gpu1[idx].gray;
        color_gpu[idx].rgba = ((uint32_t)g << 16) | ((uint32_t)g << 8) | (uint32_t)g;
        return;
    }

    int no = gray_gpu1[(y-1)*width + (x-1)].gray;
    int n  = gray_gpu1[(y-1)*width + (x)].gray;
    int ne = gray_gpu1[(y-1)*width + (x+1)].gray;

    int o  = gray_gpu1[(y)*width + (x-1)].gray;
    int e  = gray_gpu1[(y)*width + (x+1)].gray;

    int so = gray_gpu1[(y+1)*width + (x-1)].gray;
    int s  = gray_gpu1[(y+1)*width + (x)].gray;
    int se = gray_gpu1[(y+1)*width + (x+1)].gray;

    float dx = (ne - no) + 2 * (e - o) + (se - so);
    float dy = (se - ne) + 2 * (s - n) + (so - no);

    float val = sqrtf(dx * dx + dy * dy) / 4.0f;

    uint8_t g = (val > 50.0f) ? 255 : 0;

    uint32_t rgb = (g<<16) | (g<<8) | g;

    color_gpu[idx].rgba = rgb;
}



// CPU

void allocate_gpu_buffers(size_t max_pixel, packed_pixel** color_image_gpu, gray_pixel** gray_image_gpu1, gray_pixel** gray_image_gpu2)
{
    checkCudaErrors(cudaMalloc((void**)color_image_gpu, max_pixel * sizeof(packed_pixel)));
    checkCudaErrors(cudaMalloc((void**)gray_image_gpu1, max_pixel * sizeof(gray_pixel)));
    checkCudaErrors(cudaMalloc((void**)gray_image_gpu2, max_pixel * sizeof(gray_pixel)));
}

void send_data_to_gpu(animated_gif *image, int index, packed_pixel* color_image_gpu)
{
    const size_t pixel_count = (size_t)image->width[index] * image->height[index];
    packed_pixel* color_packed_image_cpu = (packed_pixel*)malloc(pixel_count * sizeof(packed_pixel));

    const pixel* color_image_cpu = image->p[index];

    for(int pixel_idx = 0; pixel_idx < pixel_count; pixel_idx++)
    {
        const uint8_t r = (uint8_t) color_image_cpu[pixel_idx].r;
        const uint8_t g = (uint8_t) color_image_cpu[pixel_idx].g;
        const uint8_t b = (uint8_t) color_image_cpu[pixel_idx].b;

        const uint32_t rgb = ((uint32_t)r << 16) | ((uint32_t)g << 8) | ((uint32_t)b);
        
        color_packed_image_cpu[pixel_idx].rgba = rgb;
    }

    checkCudaErrors(cudaMemcpy(color_image_gpu, color_packed_image_cpu, pixel_count * sizeof(packed_pixel), cudaMemcpyHostToDevice));

    free(color_packed_image_cpu);
}

void apply_gray_filter(animated_gif *image, int index, packed_pixel* color_image_gpu, gray_pixel* gray_image_gpu)
{
    size_t pixel_count = (size_t)image->width[index] * image->height[index];
    
    int block = 256;
    int grid = (int)((pixel_count + block - 1) / block);

    gray_kernel<<<grid, block>>>(color_image_gpu, gray_image_gpu, (int)pixel_count);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void apply_blur_filter(animated_gif *image, int index, int size, int threshold, gray_pixel* gray_in, gray_pixel* gray_temp) 
{
    int width = image->width[index];
    int height = image->height[index];

    gray_pixel* in = gray_in;
    gray_pixel* out = gray_temp;

    dim3 block(16,16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    uint8_t end = 0;
    uint8_t* gpu_end_flag = NULL;
    checkCudaErrors(cudaMalloc((void**)&gpu_end_flag, sizeof(*gpu_end_flag)));

    do
    {
        checkCudaErrors(cudaMemset(gpu_end_flag, 1, sizeof(*gpu_end_flag)));

        blur_kernel<<<grid, block>>>(in, out, width, height, size);
        checkCudaErrors(cudaGetLastError());

        blur_check_kernel<<<grid, block>>>(in, out, width, height, threshold, gpu_end_flag);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(&end, gpu_end_flag, sizeof(*gpu_end_flag), cudaMemcpyDeviceToHost));
        printf("end = %i\n", end);
        if(end == 1) { break; }

        gray_pixel* tmp = in;
        in = out;
        out = tmp;
    }
    while(threshold > 0);

    if (in != gray_in) 
    {
        size_t pixel_count = (size_t)width * height;
        checkCudaErrors(cudaMemcpy(gray_in, in, pixel_count * sizeof(gray_pixel), cudaMemcpyDeviceToDevice));
    }

    checkCudaErrors(cudaFree(gpu_end_flag));

}

void apply_sobel_filter(animated_gif *image, int index, gray_pixel* gray_gpu, packed_pixel* color_gpu) 
{
    int width = image->width[index];
    int height = image->height[index];

    dim3 block(16,16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    sobel_kernel<<<grid,block>>>(gray_gpu, color_gpu, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void fetch_data_from_gpu(animated_gif *image, int index, packed_pixel* color_image_gpu)
{
    int width = image->width[index];
    int height = image->height[index];
    int pixel_count = width * height;

    packed_pixel* color_packed_image_cpu = (packed_pixel*)malloc(pixel_count * sizeof(packed_pixel));
    checkCudaErrors(cudaMemcpy(color_packed_image_cpu, color_image_gpu, pixel_count * sizeof(packed_pixel), cudaMemcpyDeviceToHost));

    pixel* color_image_cpu = image->p[index];
    for(int i = 0; i < pixel_count; i++)
    {
        uint32_t rgba = color_packed_image_cpu[i].rgba;
        color_image_cpu[i].r = (rgba >> 16) & 0xFF;
        color_image_cpu[i].g = (rgba >> 8) & 0xFF;
        color_image_cpu[i].b = rgba & 0xFF;
    }

    free(color_packed_image_cpu);
}

/*
 * Main entry point
 */
int main(int argc, char **argv) {
    char *input_filename, *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;

    /* Check command-line arguments */
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.gif output.gif \n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels(input_filename);
    if (image == NULL) {
        return 1;
    }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
           input_filename, image->n_images, duration);

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    // /* Convert the pixels into grayscale */
    // apply_gray_filter(image);

    // /* Apply blur filter with convergence value */
    // apply_blur_filter(image, 5, 20);

    // /* Apply sobel filter on pixels */
    // apply_sobel_filter(image);

    size_t max_pixels = 0;
    for (int i = 0; i < image->n_images; i++) 
    {
        size_t pixels = (size_t)image->width[i] * (size_t)image->height[i];
        if (pixels > max_pixels) { max_pixels = pixels; }
    }

    packed_pixel* color_gpu = NULL;
    gray_pixel* gray_gpu1 = NULL;
    gray_pixel* gray_gpu2 = NULL;
    allocate_gpu_buffers(max_pixels, &color_gpu, &gray_gpu1, &gray_gpu2);

    for(int i=0;i<image->n_images;i++)
    {
        printf("%i / %i\n", i, image->n_images);

        send_data_to_gpu(image, i, color_gpu);

        apply_gray_filter(image,i,color_gpu,gray_gpu1);

        apply_blur_filter(image,i,5,20,gray_gpu1,gray_gpu2);


        apply_sobel_filter(image,i,gray_gpu1,color_gpu);

        fetch_data_from_gpu(image, i, color_gpu);
    }

    cudaFree(color_gpu);
    cudaFree(gray_gpu1);
    cudaFree(gray_gpu2);

    



    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("SOBEL done in %lf s\n", duration);

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if (store_pixels(output_filename, image)) {
        return 1;
    }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("Export done in %lf s in file %s\n", duration, output_filename);

    return 0;
}
