#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE 32

#define N 512
#define ROWS_A N
#define COLLUMS_A N

#define ROWS_B N
#define COLLUMS_B N

#define VECTOR_SIZE_A COLLUMS_A * ROWS_A
#define VECTOR_SIZE_B COLLUMS_B * ROWS_B
#define VECTOR_SIZE_C ROWS_A * COLLUMS_B

#define MAX_THRESHOLD  0.0001

int gemm(double** a, double** b, double** c, long lin_a, long col_a, long lin_b,
		long col_b) {
	long i, j, k;
	if (col_a != lin_b)
		return -1;
	for (i = 0; i < lin_a; i++)
		for (j = 0; j < col_b; j++) {
			c[i][j] = 0;
			for (k = 0; k < col_a; k++)
				c[i][j] += a[i][k] * b[k][j];
		}
	return 0;
}

int gemm_1d(double* a, double* b, double* c, long lin_a, long col_a, long lin_b,
		long col_b, long col_c, long lin_c) {
	long i, j, k;
	if (col_a != lin_b)
		return -1;

	for (i = 0; i < lin_a; i++) {
		for (j = 0; j < col_b; j++) {
			long index_c = i * col_c + j;
			c[index_c] = 0;
			for (k = 0; k < col_a; k++) {
				c[index_c] += a[i * col_a + k] * b[k * col_b + j];
			}
		}
		//printf("\n");
	}
	return 0;
}

__global__ void mat_cpy(double *dst, double *src, long collums, long rows) {
	long x = (blockDim.x * blockIdx.x) + threadIdx.x;
	long y = (blockDim.y * blockIdx.y) + threadIdx.y;

	long index = (collums * y) + x;

	if (collums * rows > index)
		dst[index] = src[index];
}


//since dgemm is optimized for square matrices
__global__ void calc_abft_values(double *a, double *b, long collums,
		long rows) {
	long i = blockIdx.y * blockDim.y + threadIdx.y;
	long j = blockIdx.x * blockDim.x + threadIdx.x;
	//first ABRAHAM operation
	__shared__ double acc = 0;
	__syncthreads();

	attomicAdd(&acc, a[i * collums + j]);
	__syncthreads();

	a[rows - 1 + j] = acc;
}


__global__ void mat_mult(double *dst, double *a, double *b, long size) {
	long row = blockIdx.y * blockDim.y + threadIdx.y;
	long col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row > size || col > size)
		return;

	double acc = 0;
	long index_dst = row * N + col;
	long k;
	for (k = 0; k < size; k++) {
		acc += a[row * size + k] * b[k * size + col];
	}
	dst[index_dst] = acc;
}

void print_mat(double *mat, long n, long m, const char *mat_name) {
	printf("printing %s\n", mat_name);
	long i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++)
			printf("%ld ", (long) mat[i * n + j]);
		printf("\n");
	}
}

void fill_mat(double* t, long n) {
	long i;
	for (i = 0; i < n; i++) {
		t[i] = i;
	}
}

void compare(double *t, double *s, long siz) {
	long i;
	for (i = 0; i < siz; i++) {
		if (fabs(t[i]) - fabs(s[i]) > 0.0000001)
			printf("t[%ld] is diff from s[%ld] on diff %lf", i, i,
					fabs(t[i]) - fabs(s[i]));
	}
}

void matrix_multiplication_no_abft() {
	const long siz_a = VECTOR_SIZE_A * sizeof(double);
	const long siz_b = VECTOR_SIZE_B * sizeof(double);
	const long siz_c = VECTOR_SIZE_C * sizeof(double);
	//host memories
	double* host_array_a = (double*) calloc(VECTOR_SIZE_A, sizeof(double));
	double* host_array_b = (double*) calloc(VECTOR_SIZE_B, sizeof(double));
	double* host_array_c = (double*) calloc(VECTOR_SIZE_C, sizeof(double));
	double* host_array_c_temp = (double*) calloc(VECTOR_SIZE_C, sizeof(double));
	fill_mat(host_array_a, VECTOR_SIZE_A);
	fill_mat(host_array_b, VECTOR_SIZE_B);
	//print_mat(host_array_a, COLLUMS_A, ROWS_A, "matrix A");
	printf("\n");
	//print_mat(host_array_b, COLLUMS_B, ROWS_B, "matrix B");
	//perform host matrix multiplication
	//	gemm_1d(host_array_a, host_array_b, host_array_c_temp, ROWS_A, COLLUMS_A,
	//			ROWS_B, COLLUMS_B, ROWS_A, COLLUMS_B);
	//print_mat(host_array_c_temp, COLLUMS_B, ROWS_A, "matrix C temp");
	//cuda memories
	double *device_array_a, *device_array_b, *device_array_c;
	cudaMalloc(&device_array_a, siz_a);
	cudaMalloc(&device_array_b, siz_b);
	cudaMalloc(&device_array_c, siz_c);
	//copy to device
	cudaMemcpy(device_array_a, host_array_a, siz_a, cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_b, host_array_b, siz_b, cudaMemcpyHostToDevice);
	//kernel parameters
	//we know that each block has 1024 threads
	long blocks = ceil(N / float(BLOCK_SIZE));
	long threads = ceil(N / float(blocks));
	//2d grid
	dim3 gridDim(blocks, blocks);
	//threads num, 2d
	dim3 blockDim(threads, threads);
	mat_mult<<<gridDim, blockDim>>>(device_array_c, device_array_a,
			device_array_b, N);
	printf("\nblocks %ld threads %ld\n", blocks, threads);
	cudaMemcpy(host_array_c, device_array_c, siz_c, cudaMemcpyDeviceToHost);
	//print_mat(host_array_c, COLLUMS_A, ROWS_A, "GPU result mat");
	printf("compare matrices\n");
	//compare(host_array_c, host_array_c_temp, VECTOR_SIZE_C);
	cudaFree(device_array_a);
	cudaFree(device_array_b);
	cudaFree(device_array_c);
	free(host_array_a);
	free(host_array_b);
	free(host_array_c);
	free(host_array_c_temp);
}

void matrix_multiplication_abft() {
	//first matrix has rows + 1
	const long a_rows = ROWS_A + 1;
	const long a_coll = COLLUMS_A + 1;
	const long b_rows = ROWS_B + 1;
	const long b_coll = COLLUMS_B + 1;
	//total size of all vectors
	const long a_pure_siz = (a_rows * a_coll);
	const long b_pure_siz = (b_rows * b_coll);
	const long c_pure_siz = (a_rows * b_coll);
	//---------------------------
	//call byte size
	const long siz_a = a_pure_siz * sizeof(double);
	const long siz_b = b_pure_siz * sizeof(double);
	const long siz_c = (a_rows * b_coll) * sizeof(double);
	//--------------------------
	//allocate all host memory
	double* host_array_a = (double*) calloc(a_pure_siz, sizeof(double));
	double* host_array_b = (double*) calloc(b_pure_siz, sizeof(double));
	double* host_array_c = (double*) calloc(c_pure_siz, sizeof(double));
	double* host_array_c_gpu = (double*) calloc(c_pure_siz, sizeof(double));
	//fill a and b matrices
	fill_mat(host_array_a, a_pure_siz);
	fill_mat(host_array_b, b_pure_siz);

	//cuda memories
	double *device_array_a, *device_array_b, *device_array_c;
	cudaMalloc(&device_array_a, siz_a);
	cudaMalloc(&device_array_b, siz_b);
	cudaMalloc(&device_array_c, siz_c);

	//copy memory to device
	cudaMemcpy(device_array_a, host_array_a, siz_a, cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_b, host_array_b, siz_b, cudaMemcpyHostToDevice);

	//

	free(host_array_a);
	free(host_array_b);
	free(host_array_c);
	free(host_array_c_gpu);
	cudaFree(device_array_a);
	cudaFree(device_array_b);
	cudaFree(device_array_c);
}

int main(void) {

	matrix_multiplication_abft();
	return 0;
}
