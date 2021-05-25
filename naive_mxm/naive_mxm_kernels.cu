#include <iostream>
#include <string>

#include "helper_cuda.h"

#define BLOCK_SIZE 32

void _checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	std::cout << "CUDA Framework error: " << cudaGetErrorString(error);
	std::cout << "Bailing. " << "Line: " << line << " file: " << file
			<< std::endl;
	exit(EXIT_FAILURE);
}

#define checkFrameworkErrors(error) _checkFrameworkErrors(error, __LINE__, __FILE__)

typedef struct {
	int *rows;
	int *cols;
	float *values;
} sparse_mat;

__global__
void naive_common_mxm(float* a, float* b, float *c, int n) {
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	float acc = 0.0;
	for (int k = 0; k < n; k++) {
		acc += a[ty * n + k] * b[k * n + tx];
	}
	c[ty * n + tx] = acc;
}


int main(int argc, char** argv) {
	std::cout << "Naive MXM\n";
	int n = 32;
	int size = n * n;
	sparse_mat a, b, c;
	malloc_sparse_matrix(a, size);
	malloc_sparse_matrix(b, size);
	malloc_sparse_matrix(c, size);

//	naive_sparse_mxm<<<32, 32>>>(a, b, c, 32);
	checkFrameworkErrors(cudaDeviceSynchronize());

	free_sparse_matrix(a);
	free_sparse_matrix(b);
	free_sparse_matrix(c);
}
