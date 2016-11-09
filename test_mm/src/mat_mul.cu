#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE 32

#define ROWS_A 4
#define COLLUMS_A 6

#define ROWS_B 6
#define COLLUMS_B 4

#define VECTOR_SIZE_A COLLUMS_A * ROWS_A
#define VECTOR_SIZE_B COLLUMS_B * ROWS_B
#define VECTOR_SIZE_C ROWS_A * COLLUMS_B

int gemm(double** a, double** b, double** c, long lin_a, long col_a, long lin_b, long col_b){
    long i, j, k;
    if (col_a != lin_b)
        return -1;
    for (i = 0; i < lin_a; i++)
        for (j = 0; j < col_b; j++){
            c[i][j] = 0;
            for (k = 0; k < col_a; k++)
                c[i][j] += a[i][k] * b[k][j];
        }
    return 0;
}


__global__ void mat_cpy(double *dst, double *src, long collums, long rows){
	long x = (blockDim.x * blockIdx.x) + threadIdx.x;
	long y = (blockDim.y * blockIdx.y) + threadIdx.y;

	long index = (collums * y) + x;

	if(collums * rows > index)
		dst[index] = src[index];
}


__global__ void mat_mult(double *dst, double *a, double *b, long col_a, long col_b, long row_a, long row_b){
	long x = (blockDim.x * blockIdx.x) + threadIdx.x;
	long y = (blockDim.y * blockIdx.y) + threadIdx.y;

	long c_index = (col_b * y) + x;
	double acc = 0;
	long k;
	printf("%d %d\n", x, y);
	for(k = 0; k < row_a; k++)
		acc += a[x * col_a + k] + b[k * col_b + y];

	dst[c_index] = acc;

}


void print_mat(double *mat, long n, long m, const char *mat_name){
	printf("printing %s\n", mat_name);
	long i, j;
	for(i = 0; i < m;i++){
		for(j = 0; j < n; j++) printf("%d ", (int)mat[i * n + j]);
		printf("\n");
	}
}

void fill_mat(double* t, long n){
    long i;
    for (i = 0; i < n; i++){
            t[i] = 1; //(double) random() / (double)30000;
    }
}

int main(void) {

	const long siz_a = VECTOR_SIZE_A * sizeof(double);
	const long siz_b = VECTOR_SIZE_B * sizeof(double);
	const long siz_c = VECTOR_SIZE_C * sizeof(double);
	//host memories
	double* host_array_a = (double*)calloc(VECTOR_SIZE_A,sizeof(double));
	double* host_array_b = (double*)calloc(VECTOR_SIZE_B,sizeof(double));
	double* host_array_c = (double*)calloc(VECTOR_SIZE_C,sizeof(double));
	fill_mat(host_array_a, VECTOR_SIZE_A);
	fill_mat(host_array_b, VECTOR_SIZE_B);

	print_mat(host_array_a, COLLUMS_A, ROWS_A, "matrix A");
	printf("\n");
	print_mat(host_array_b, COLLUMS_B, ROWS_B, "matrix B");

	//cuda memories
	double *device_array_a, *device_array_b, *device_array_c;
	cudaMalloc(&device_array_a, siz_a);
	cudaMalloc(&device_array_b, siz_b);
	cudaMalloc(&device_array_c, siz_c);

	//copy to device
	cudaMemcpy(device_array_a, host_array_a, siz_a, cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_b, host_array_b, siz_b, cudaMemcpyHostToDevice);

	//kernel parameters
	//we know that each block has 512 threads
	//so
	long threadsPerBlock = min(COLLUMS_A, BLOCK_SIZE);
	long blocks  = ceilf((double) COLLUMS_A / (double) threadsPerBlock);

	//printf("%ld %ld %ld\n", threadsPerBlock, blocks, threads);
	dim3 blockDim(blocks, blocks, 1);

	dim3 threaDim(ROWS_A, COLLUMS_B, 1);
	printf("blocks %ld threads %ld\n", blocks, threaDim.x);

	mat_mult<<<blockDim, threaDim>>>(device_array_c, device_array_a, device_array_b, COLLUMS_A, COLLUMS_B, ROWS_A, ROWS_B);

	//printf("\nVECTOR SIZE %d\n", VECTOR_SIZE);
	cudaMemcpy(host_array_c, device_array_c, siz_c, cudaMemcpyDeviceToHost);
	print_mat(host_array_c, COLLUMS_A, ROWS_A, "result mat");


	cudaFree(device_array_a);
	cudaFree(device_array_b);
	cudaFree(device_array_c);
	free(host_array_a);
	free(host_array_b);
	free(host_array_c);

	return 0;
}
