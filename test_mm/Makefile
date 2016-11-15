NVCC = nvcc
CC = /usr/bin/gcc
CXX = /usr/bin/g++
SRC=src/

VEC_ADD = vec_add
MAT_MUL = mat_mul

GENCODE=-gencode arch=compute_35,code=\'compute_35,sm_35\'

NVCCFLAGS = -L/usr/local/cuda-7.0/lib64/ -lcublas -lcublas_device -lcudadevrt -lcudart -rdc=true
#-lcublas
ifeq ($(DEBUG), 1)
NVCCFLAGS += -G -g

endif

NVCC_CDP = 

TARGET = $(VEC_ADD) $(MAT_MUL)

all: clean $(MAT_MUL)

$(MAT_MUL):$(SRC)$(MAT_MUL).cu
	$(NVCC) $(SRC)$(MAT_MUL).cu -o $(MAT_MUL) $(NVCCFLAGS) $(GENCODE)
	

clean:
	rm -f $(TARGET) *.o
