#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE 32

#define N 6
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

//since dgemm is optimized for square matrices I'm going to use
//first ABRAHAM operation
//	for (j = 0; j < col_a; j++) {
//		acc = 0;
//		for (i = 0; i < lin_a; i++)
//
//			acc += a[i * col_a + j];
//
//        a[lin_a * col_a + j] = acc;
//	}
__global__ void first_abraham_op(double *a, long collums,
		long rows) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;
	//iterate on j dimension
	long i;
	double acc = 0;
	for (i = 0; i < rows; i++) {
		acc += a[i * collums + j];
	}

	a[rows * collums + j] = acc;
}

/**
 * 	for (i = 0; i < lin_b; i++) {
		acc = 0;
		for (j = 0; j < col_b; j++)
			acc += b[i * (col_b + 1) + j];
		//printf("i * col_b %ld col b %ld  acc %lf\n", i * col_b, col_b, acc);
		b[i * (col_b + 1) + col_b] = acc;
	}
 */
__global__ void second_abraham_op(double *b, long collums, long rows){
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	long j;
	double acc = 0;
	for(j = 0; j < collums; j++){
		acc += b[i * (collums + 1) + j];
	}

	b[i * (collums + 1) + collums] = acc;
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

void print_mat(double *mat, long m, long n, const char *mat_name) {
	printf("printing %s lin %ld col %ld\n", mat_name, m, n);
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
		t[i] = 1;
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

int gemm_ongpu_abft(double *a, double *b, double *c, long lin_a, long col_a,
		long lin_b, long col_b) {
	long i, j;
	double acc = 0;
	int ret = 0;
	long col_c = col_b;
	long lin_c = lin_a;
	//first ABRAHAM operation
	for (j = 0; j < col_a; j++) {
		acc = 0;
		for (i = 0; i < lin_a; i++)

			acc += a[i * col_a + j];

        a[lin_a * col_a + j] = acc;
	}

	//second ABRAHAM operation
	for (i = 0; i < lin_b; i++) {
		acc = 0;
		for (j = 0; j < col_b; j++)
			acc += b[i * (col_b + 1) + j];
		//printf("i * col_b %ld col b %ld  acc %lf\n", i * col_b, col_b, acc);
		b[i * (col_b + 1) + col_b] = acc;
	}

	//print_mat(a, lin_a + 1, col_a);
	//printf("\n");
	//print_mat(b, lin_b, col_b + 1);
	//performs matrix multiplication
	gemm_1d(a, b, c, lin_a + 1, col_a, lin_b, col_b + 1, col_b + 1, lin_a + 1);

	//check all checksums
	//line checksum
	for (j = 0; j < col_a; j++) {
		acc = 0;
		for (i = 0; i < lin_a; i++)
			acc += c[i * col_c + j];

		if (fabs(c[lin_a * col_c + j]) - fabs(acc) >= MAX_THRESHOLD) {
//			printf(
//					"lin - position corrupted [%ld][%ld] - exp chsum %lf got chsum %lf diff - %lf\n",
//					lin_a, j, c[lin_a * col_c + j], acc,
//					c[lin_a * col_c + j] - acc);
			ret++;
		}
	}

	//collum checksum
	for (i = 0; i < lin_b; i++) {
		acc = 0;
		for (j = 0; j < col_b; j++)
			acc += c[i * col_c + j];

		if (fabs(c[i * col_c + col_b] - acc) >= MAX_THRESHOLD) {
//			printf(
//					"collum - position corrupted [%ld][%ld] - exp chsum %lf got chsum %lf diff %lf\n",
//					i, col_b, c[i * col_c + col_b], acc,
//					c[i * col_c + col_b] - acc);
			ret++;
		}
	}
	return ret;

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

void matrix_multiplication_abft(){
	long size = 10;
	long lin_a = 10;
	long col_a = 10;
	long lin_b = col_a;
	long col_b = 10;
	long vec_siz_a = ((lin_a + 1) * col_a);
	long vec_siz_b = (lin_b * (col_b + 1));
	long vec_siz_c = ((lin_a + 1) * (col_b + 1));
	const long siz_a = vec_siz_a * sizeof(double);
	const long siz_b = vec_siz_b * sizeof(double);
	const long siz_c = vec_siz_b * sizeof(double);
	//host memories
	double* host_array_a = (double*) calloc(vec_siz_a, sizeof(double));
	double* host_array_b = (double*) calloc(vec_siz_b, sizeof(double));
	double* host_array_c = (double*) calloc(vec_siz_c, sizeof(double));
	double* host_array_c_temp = (double*) calloc(vec_siz_c, sizeof(double));
	fill_mat(host_array_a, vec_siz_a);
	fill_mat(host_array_b, vec_siz_b);

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
	//these vars are for mat multplication,
	long blocks = ceil(size / float(BLOCK_SIZE));
	long threads = ceil(size / float(blocks));

	//2d grid
	dim3 gridDim(blocks, blocks);
	//threads num, 2d
	dim3 blockDim(threads, threads);

	//1d grid for abft operations
	long threads_abft_first = ceil(lin_a / float(blocks));
	long threads_abft_second = ceil(col_b / float(blocks));

	first_abraham_op<<<blocks, threads_abft_first>>>(device_array_a, lin_a, col_a);
	second_abraham_op<<<blocks, threads_abft_second>>>(device_array_b, lin_b, col_b);
	cudaMemcpy(host_array_a, device_array_a, siz_a, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array_b, device_array_b, siz_b, cudaMemcpyDeviceToHost);
	print_mat(host_array_a, lin_a + 1, col_a, "matrix A");
	printf("\n");
	print_mat(host_array_b, lin_b, col_b + 1, "matrix B");
	mat_mult<<<gridDim, blockDim>>>(device_array_c, device_array_a,
			device_array_b, N);
	printf("\nblocks %ld threads %ld\n", blocks, threads);
	cudaMemcpy(host_array_c, device_array_c, siz_c, cudaMemcpyDeviceToHost);
	print_mat(host_array_c, lin_a  +1 , col_b +1, "GPU result mat");
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



int main(void) {
//	long m_a = 15;
//	long n_a = 10;
//	long m_b = n_a;
//	long n_b = 12;
//	double a[(m_a + 1) * n_a], b[m_a * (n_a + 1)], c[(m_a + 1) * (n_b + 1)];
//
//	fill_mat(a, (m_a + 1) * n_a);
//	fill_mat(b, m_b * (n_b + 1));
//
////	print_mat(a, m_a + 1,  n_a,  "matrix a");
////	print_mat(b, m_b, n_b + 1, "matrix b");
//
//	gemm_ongpu_abft(a, b, c, m_a, n_a, m_b, n_b);
//	print_mat(a, m_a + 1,  n_a,  "matrix a");
//	print_mat(b, m_b, n_b + 1, "matrix b");
//	print_mat(c, m_a + 1, n_b + 1, "matrix c");

	matrix_multiplication_abft();
	return 0;
}
