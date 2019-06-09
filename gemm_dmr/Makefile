CXX=g++
CXXFLAGS= -std=c++11 -o3  -fPIC 
EXEC=gemmDmr
LOGS=0

VPATH=./
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc


NVCCFLAGS= -std=c++11 -O3 -Xptxas -v

ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
ARCH+= -gencode arch=compute_62,code=[sm_62,compute_62] #Tegra X2
#ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72] #XavierV
INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc

#read_only_cache_kernels.o
OBJDIR=./obj/
OBJ= main.o


ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
LDFLAGS+= -L../../include  -lLogHelper
endif


LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt -lcublas -lpthread

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile


all: mkdir $(EXEC)


$(EXEC): $(OBJS)  
	$(CXX) $^ -o $@ $(INCLUDE) $(LDFLAGS) 

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE) 
	

mkdir:
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)
	
	
test:
	./$(EXEC) -dimX=16384 -dimY=16384

