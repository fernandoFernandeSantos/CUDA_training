NVCC = nvcc
CC = /usr/bin/gcc
CXX = /usr/bin/g++
SRC=src/

VEC_ADD = vec_add
MAT_MUL = mat_mul

NVCCFLAGS =   -lcublas -lcublas_device -lcudadevrt -gencode arch=compute_35,code=sm_35 -lcudart -rdc=true -dc
ifeq ($(DEBUG), 1)
NVCCFLAGS += -G -g

endif

NVCC_CDP = 

TARGET = $(VEC_ADD) $(MAT_MUL)

all: $(VEC_ADD) $(MAT_MUL)

$(VEC_ADD):$(SRC)$(VEC_ADD).cu
	$(NVCC) $(NVCCFLAGS) $(SRC)$(VEC_ADD).cu -o $(VEC_ADD)

$(MAT_MUL):$(SRC)$(MAT_MUL).cu
	$(NVCC) $(SRC)$(MAT_MUL).cu -o $(MAT_MUL).o $(NVCCFLAGS)
	
clean:
	rm -f $(TARGET) *.o

