SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
NVCC=nvcc

CFLAGS=-O3 -fopenmp -I$(HEADER_DIR)
NVFLAGS=-O3 -I$(HEADER_DIR)

LDFLAGS=-lm
CUDA_LDFLAGS=-lm

TARGETS = sobelf_seq sobelf_mpi sobelf_omp sobelf_hyb sobelf_cuda

SRCS_COMMON = dgif_lib egif_lib gif_err gif_font gif_hash \
			  gifalloc gif_io openbsd-reallocarray quantize
OBJ_COMMON = $(SRCS_COMMON:%=$(OBJ_DIR)/%.o)

all: $(OBJ_DIR) $(TARGETS)

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

$(OBJ_DIR)/main_cuda.o: $(SRC_DIR)/main_cuda.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $<

sobelf_%:$(OBJ_COMMON) $(OBJ_DIR)/main_%.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

sobelf_cuda: $(OBJ_COMMON) $(OBJ_DIR)/main_cuda.o
	$(NVCC) $(NVFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

clean:
	rm -f $(TARGETS) $(OBJ_DIR)/*.o
