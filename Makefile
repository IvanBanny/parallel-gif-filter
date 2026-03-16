SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
NVCC=nvcc

CFLAGS=-O3 -fopenmp -I$(HEADER_DIR) -Wall -Wextra
NVFLAGS=-O3 -I$(HEADER_DIR) -Xcompiler -Wall -Xcompiler -Wextra -diag-suppress 541

LDFLAGS=-lm
CUDA_LDFLAGS=-lm

TARGETS = sobelf_seq sobelf_mpi sobelf_omp sobelf_hyb sobelf_cuda

SRCS_LIB = dgif_lib egif_lib gif_err gif_font gif_hash \
		   gifalloc openbsd-reallocarray quantize
OBJ_LIB = $(SRCS_LIB:%=$(OBJ_DIR)/%.o)
OBJ_COMMON = $(OBJ_LIB) $(OBJ_DIR)/gif_io.o

all: $(OBJ_DIR) $(TARGETS)

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_LIB): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) -O3 -fopenmp -I$(HEADER_DIR) -c -o $@ $^

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
