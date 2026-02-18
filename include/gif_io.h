#ifndef GIF_IO_H
#define GIF_IO_H

#include "gif_lib.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

typedef struct pixel {
    int r;
    int g;
    int b;
} pixel;

typedef struct animated_gif {
    int n_images;
    int *width;
    int *height;
    pixel **p;
    GifFileType *g;
} animated_gif;

#define CONV(l, c, nb_c) ((l) * (nb_c) + (c))

animated_gif *load_pixels(char *filename);
int output_modified_read_gif(char *filename, GifFileType *g);
int store_pixels(char *filename, animated_gif *image);

#endif
