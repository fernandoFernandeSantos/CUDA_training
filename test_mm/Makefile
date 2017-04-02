NVCC = nvcc
CC = /usr/bin/gcc
CXX = /usr/bin/g++
SRC=src/

VEC_ADD = vec_add
MAT_MUL = mat_mul

GENCODE=-gencode arch=compute_35,code=sm_35

NVCCFLAGS = -L/usr/local/cuda-8.0/lib64/ -lcublas -lcublas_device -lcudadevrt -lcudart -rdc=true -I/usr/local/cuda-8.0/samples/common/inc/
#-lcublas
ifeq ($(DEBUG), 1)
NVCCFLAGS += -G -g

endif

NVCC_CDP = 

TARGET = $(VEC_ADD) $(MAT_MUL)

all: clean $(MAT_MUL)

$(MAT_MUL):$(SRC)$(MAT_MUL).cu
	$(NVCC) $(SRC)abft.cu $(SRC)$(MAT_MUL).cu -o $(MAT_MUL) $(NVCCFLAGS) $(GENCODE) $(FI)

sgemv:
	$(NVCC) $(SRC)sgemv_compare.cu -o sgemv $(NVCCFLAGS) $(GENCODE) $(FI)	

clean:
	rm -f $(TARGET) *.o
