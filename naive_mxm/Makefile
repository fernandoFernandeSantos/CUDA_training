CXX = g++-8
TARGET=naive_mxm

CUDAPATH = /usr/local/cuda
NVCC = $(CUDAPATH)/bin/nvcc

ARCH = -gencode arch=compute_35,code=sm_35

NVCCFLAGS = -L$(CUDAPATH)/lib64/ -lcuda -std=c++11 -I$(CUDAPATH)/samples/common/inc

ifeq ($(DEBUG), 1)
NVCCFLAGS += -G -g
endif


all: clean $(TARGET)


$(TARGET):
	$(NVCC) -ccbin $(CXX) $(ARCH) naive_mxm_kernels.cu -o $(TARGET)  $(NVCCFLAGS)

clean:
	rm -f $(TARGET) *.o
