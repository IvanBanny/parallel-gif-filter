SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
CFLAGS=-O3 -fopenmp -I$(HEADER_DIR)
LDFLAGS=-lm

OBJ_COMMON= $(OBJ_DIR)/dgif_lib.o \
	$(OBJ_DIR)/egif_lib.o \
	$(OBJ_DIR)/gif_err.o \
	$(OBJ_DIR)/gif_font.o \
	$(OBJ_DIR)/gif_hash.o \
	$(OBJ_DIR)/gifalloc.o \
	$(OBJ_DIR)/gif_io.o \
	$(OBJ_DIR)/openbsd-reallocarray.o \
	$(OBJ_DIR)/quantize.o

OBJ_SEQ= $(OBJ_COMMON) $(OBJ_DIR)/main_seq.o
OBJ_MPI= $(OBJ_COMMON) $(OBJ_DIR)/main_mpi.o
OBJ_OMP= $(OBJ_COMMON) $(OBJ_DIR)/main_omp.o

all: $(OBJ_DIR) sobelf_seq sobelf_mpi sobelf_omp

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

sobelf_seq:$(OBJ_SEQ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

sobelf_mpi: $(OBJ_MPI)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

sobelf_omp: $(OBJ_OMP)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f sobelf_seq sobelf_mpi sobelf_omp $(OBJ_SEQ) $(OBJ_DIR)/main_mpi.o $(OBJ_DIR)/main_omp.o


