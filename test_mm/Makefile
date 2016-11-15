NVCC = nvcc
CC = /usr/bin/gcc
CXX = /usr/bin/g++
SRC=src/

VEC_ADD = vec_add
MAT_MUL = mat_mul

GENCODE=-gencode arch=compute_35,code=sm_35

NVCCFLAGS = -lcublas -lcublas_device -lcudadevrt -lcudart -rdc=true -dc 
#-lcublas
ifeq ($(DEBUG), 1)
NVCCFLAGS += -G -g

endif

NVCC_CDP = 

TARGET = $(VEC_ADD) $(MAT_MUL)

all: $(MAT_MUL)

$(MAT_MUL):$(SRC)$(MAT_MUL).cu
	$(NVCC) $(SRC)$(MAT_MUL).cu -o $(MAT_MUL).o $(NVCCFLAGS) $(GENCODE)
	$(NVCC) $(MAT_MUL).o -o $(MAT_MUL) -lcublas -gencode arch=compute_35,code=sm_35
	
clean:
	rm -f $(TARGET) *.o
