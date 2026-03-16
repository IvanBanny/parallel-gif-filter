SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
NVCC=nvcc

MPI_INC = $(shell mpicc --showme:compile | grep -o '\-I[^ ]*')
MPI_LIBDIRS = $(shell mpicc --showme:libdirs | sed 's/[^ ]* */-L&/g')
MPI_LIBS = $(shell mpicc --showme:libs | sed 's/[^ ]* */-l&/g')

CFLAGS=-O3 -fopenmp -I$(HEADER_DIR) -Wall -Wextra
NVFLAGS=-O3 -I$(HEADER_DIR) -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fopenmp -diag-suppress 541

ifdef SKIP_EXPORT
    CFLAGS += -DSKIP_EXPORT
    NVFLAGS += -DSKIP_EXPORT
endif

LDFLAGS=-lm
CUDA_LDFLAGS=-lm

TARGETS = sobelf_seq sobelf_mpi sobelf_omp sobelf_hyb sobelf_cuda sobelf_all

SRCS_LIB = dgif_lib egif_lib gif_err gif_font gif_hash \
		   gifalloc openbsd-reallocarray quantize
OBJ_LIB = $(SRCS_LIB:%=$(OBJ_DIR)/%.o)
OBJ_COMMON = $(OBJ_LIB) $(OBJ_DIR)/gif_io.o

all: $(OBJ_DIR) $(TARGETS)

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

# libraries
$(OBJ_LIB): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) -O3 -fopenmp -I$(HEADER_DIR) -c -o $@ $^

$(OBJ_DIR)/gif_io.o: $(SRC_DIR)/gif_io.c
	$(CC) $(CFLAGS) -c -o $@ $<

# compilation

$(OBJ_DIR)/main_seq.o: $(SRC_DIR)/main_seq.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/main_mpi.o: $(SRC_DIR)/main_mpi.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/main_omp.o: $(SRC_DIR)/main_omp.c
	$(CC) $(CFLAGS) -o $@ -c $<

$(OBJ_DIR)/main_hyb.o: $(SRC_DIR)/main_hyb.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/main_cuda.o: $(SRC_DIR)/main_cuda.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $<

$(OBJ_DIR)/main_all.o: $(SRC_DIR)/main_all.cu
	$(NVCC) $(NVFLAGS) $(MPI_INC) -c -o $@ $<

#linking

sobelf_seq: $(OBJ_COMMON) $(OBJ_DIR)/main_seq.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

sobelf_mpi: $(OBJ_COMMON) $(OBJ_DIR)/main_mpi.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

sobelf_omp: $(OBJ_COMMON) $(OBJ_DIR)/main_omp.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

sobelf_hyb: $(OBJ_COMMON) $(OBJ_DIR)/main_hyb.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

sobelf_cuda: $(OBJ_COMMON) $(OBJ_DIR)/main_cuda.o
	$(NVCC) $(NVFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

sobelf_all: $(OBJ_COMMON) $(OBJ_DIR)/main_all.o
	$(NVCC) $(NVFLAGS) -o $@ $^ $(CUDA_LDFLAGS) -lgomp $(MPI_LIBDIRS) $(MPI_LIBS)

clean:
	rm -f $(TARGETS) $(OBJ_DIR)/*.o
