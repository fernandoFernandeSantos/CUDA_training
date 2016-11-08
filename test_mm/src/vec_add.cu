#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE 32

#define VECTOR_SIZE 10//1024 * 1024 * 16

__global__ void sum(double *a, double *b, double *c){
	int bx = threadIdx.x;

	c[bx] = a[bx] + b[bx];
}


int main(void) {

	//host memories
	double* host_array_a = calloc(VECTOR_SIZE,sizeof(double));
	double* host_array_b = calloc(VECTOR_SIZE,sizeof(double));
	double* host_array_c = calloc(VECTOR_SIZE,sizeof(double));
	int i;
	for(i = 0; i < VECTOR_SIZE; i++){
		host_array_a[i] = sin(i);
		host_array_b[i] = cos(i);
	}

	//cuda memories
	double *device_array_a, *device_array_b, *device_array_c;
	cudaMalloc(&device_array_a, VECTOR_SIZE);
	cudaMalloc(&device_array_b, VECTOR_SIZE);
	cudaMalloc(&device_array_c, VECTOR_SIZE);

	//copy to device
	cudaMemcpy(device_array_a, host_array_a, VECTOR_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_b, host_array_b, VECTOR_SIZE, cudaMemcpyHostToDevice);

	//kernel parameters
	//we know that each block has 512 threads
	//so
	long threadsPerBlock;
	long blocksPerGrid;
	if(VECTOR_SIZE < 512){
		threadsPerBlock = VECTOR_SIZE;
		blocksPerGrid = 1;
	}else{
		threadsPerBlock = 512;
		blocksPerGrid = ceil(double(VECTOR_SIZE)/double(threadsPerBlock));
	}


	return 0;
}
