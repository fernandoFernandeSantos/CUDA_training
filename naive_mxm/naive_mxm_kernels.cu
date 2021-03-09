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

__global__
void naive_sparse_mxm(sparse_mat a, sparse_mat b, sparse_mat c, int n) {

}

void cuda_malloc(float* x, int n) {
	checkFrameworkErrors(cudaMalloc(&x, sizeof(float) * n));
}

void cuda_free(float* x, int n) {
	checkFrameworkErrors(cudaFree(x));
}

int main(int argc, char** argv) {
	std::cout << "Naive MXM\n";
	int size = 1024 * 1024;
	sparse_mat a, b, c;
	cuda_malloc(a.values, size);

	cuda_free(a.values, size);
}
