
all: run build

build:
	/usr/local/cuda/bin/nvcc -arch=sm_70 -ccbin=g++ -o test_dmr test_dmr.cu

profile: build
	nvprof --metrics achieved_occupancy,flop_count_dp,flop_count_sp ./test_dmr 

run: build
	./test_dmr

clean:
	rm test_dmr
