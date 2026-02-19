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

OBJ= $(OBJ_COMMON) $(OBJ_DIR)/main.o
OBJ_MPI= $(OBJ_COMMON) $(OBJ_DIR)/main_mpi.o

all: $(OBJ_DIR) sobelf sobelf_mpi

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

sobelf:$(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

sobelf_mpi: $(OBJ_MPI)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f sobelf sobelf_mpi $(OBJ) $(OBJ_DIR)/main_mpi.o

