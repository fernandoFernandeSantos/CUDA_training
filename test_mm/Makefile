NVCC = /usr/local/cuda-7.0/bin/nvcc
CC = /usr/bin/gcc
CXX = /usr/bin/g++
SRC=src/

TARGET = vec_add

all: 
	$(NVCC) $(SRC)$(TARGET).cu -o $(TARGET)
	
clean:
	rm -f $(TARGET) *.o