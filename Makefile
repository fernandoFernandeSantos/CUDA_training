NVCC = nvcc
CC = /usr/bin/gcc
CXX = /usr/bin/g++
SRC=test_mm/src/

VEC_ADD = vec_add
MAT_MUL = mat_mul


GENCODE=-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_60,code=sm_60

NVCCFLAGS = -L/usr/local/cuda-8.0/lib64/ -lcublas -lcublas_device -lcudadevrt -lcudart -lcuda  -I/usr/local/cuda-8.0/samples/common/inc/
#-lcublas
ifeq ($(DEBUG), 1)
NVCCFLAGS += -G -g

endif

NVCC_CDP = 

TARGET = $(VEC_ADD) $(MAT_MUL)

all: clean $(MAT_MUL) streams streams_kernel

$(MAT_MUL):$(SRC)$(MAT_MUL).cu
	$(NVCC) --default-stream per-thread $(SRC)abft.cu $(SRC)$(MAT_MUL).cu -o $(MAT_MUL) $(NVCCFLAGS) $(GENCODE) $(FI)

streams:
	$(NVCC) --default-stream per-thread test_streams/streams.cu -o test_streams/streams $(NVCCFLAGS) $(GENCODE) 
	
streams_kernel:
		$(NVCC) --default-stream per-thread test_streams/streams_kernel.cu -o test_streams/streams_kernel $(NVCCFLAGS) $(GENCODE) 
	

clean:
	rm -f $(TARGET) *.o test_mm/mat_mul test_streams/streams_kernel test_streams/streams
