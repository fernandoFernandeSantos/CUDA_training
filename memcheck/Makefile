
ARCH= -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 \
			-gencode arch=compute_53,code=sm_53 -gencode arch=compute_61,code=sm_61 \
			-gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 \
			-gencode arch=compute_72,code=sm_72
			
CXXFLAGS= -O3 -std=c++11
NVCC=/usr/local/cuda/bin/nvcc 


all:
	$(NVCC) $(CXXFLAGS) $(ARCH) -o memcheck main.cu

