/*
 * checksum.h
 *
 *  Created on: 15/11/2016
 *      Author: fernando
 */

#ifndef CHECKSUM_H_
#define CHECKSUM_H_
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <math.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

__device__ int row_detected_errors = 0;
__device__ int col_detected_errors = 0;

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
#define PRINT_TYPE long

__global__ void check_col(double *mat, long rows, long cols) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	double acc = 0;
	//must be less one
	for (k = 0; k < cols - 1; k++) {
		acc += mat[i * cols + k];
	}
	long b_index = i * cols + cols - 1;
	//printf("b_index %ld acc %lf \n", b_index, acc);

	if (fabs(mat[b_index]) - fabs(acc)) {
		atomicAdd(&col_detected_errors, 1);
	}

}

__global__ void check_row(double *mat, long rows, long cols) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	double acc = 0;
	//must be less one
	for (k = 0; k < rows - 1; k++) {
		acc += mat[k * cols + j];
	}
	//printf("a_index %ld acc %lf \n", rows_a * cols_a + j, acc);
	long a_index = (rows - 1) * cols + j;
	if (fabs(mat[a_index]) - fabs(acc) <= MAX_THRESHOLD) {
		atomicAdd(&row_detected_errors, 1);
	}

}

//DYNAMIC PARALLELISM ONLY TO CALL NEW KERNELS, ARE FUCK KIDDING???
//man, I am so lazy
__global__ void check_checksums(double *c, long rows_c, long cols_c) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	//rows
	if (i == 0) {
		long blocks = ceil(cols_c / double(BLOCK_SIZE));
		long threads = ceil(cols_c / double(blocks));
		printf("passou no row\n");
		check_row<<<blocks, threads>>>(c, rows_c, cols_c);
	}
	//cols
	if (i == 1) {
		printf("passou no col\n");
		long blocks = ceil(rows_c / double(BLOCK_SIZE));
		long threads = ceil(rows_c / double(blocks));
		check_col<<<blocks, threads>>>(c, rows_c, cols_c);
	}
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
//rows_b MUST BE THE SAME OF cols_a
__global__ void first_abraham_op(double *a, long rows_a, long cols_a) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	double acc = 0;
	for (k = 0; k < rows_a; k++) {
		acc += a[k * cols_a + j];
	}
	//printf("a_index %ld acc %lf \n", rows_a * cols_a + j, acc);
	long a_index = (rows_a - 1) * cols_a + j;
	a[a_index] = acc;
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
__global__ void second_abraham_op(double *b, long rows_b, long cols_b) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	double acc = 0;
	for (k = 0; k < cols_b; k++) {
		acc += b[i * cols_b + k];
	}
	long b_index = i * cols_b + cols_b - 1;
	//printf("b_index %ld acc %lf \n", b_index, acc);

	b[b_index] = acc;
}

__global__ void zero_col_or_row(double *mat, long rows, long cols, long num,
		char op) {
	long p = blockIdx.x * blockDim.x + threadIdx.x;
	//zero rows
	if (op == 'r') {
		//num is which row/collum must be set to zero, 2, 3 ... or n
		mat[p * rows + num] = 0;
	} else {
		mat[rows * num + p] = 0;
	}
}

void first_abraham(double *a, long rows_a, long cols_a) {
	//1d grid for abft operations

	long blocks_abft_first = ceil((cols_a + 1) / float(BLOCK_SIZE));
	long threads_abft_first = ceil((cols_a + 1) / float(blocks_abft_first));
	(first_abraham_op<<<blocks_abft_first, threads_abft_first>>>(a, rows_a + 1,
			cols_a + 1));
	gpuErrchk( cudaPeekAtLastError() );
}

void second_abraham(double *b, long rows_b, long cols_b) {
//second
	long blocks_abft_second = ceil((rows_b + 1) / float(BLOCK_SIZE));
	long threads_abft_second = ceil((rows_b + 1) / float(blocks_abft_second));

	second_abraham_op<<<blocks_abft_second, threads_abft_second>>>(b,
			rows_b + 1, cols_b + 1);
	gpuErrchk( cudaPeekAtLastError() );
}

void abraham_check(double *c, long rows, long cols) {
	printf("passou\n");
	check_checksums<<<1, 2>>>(c, rows, cols);
	gpuErrchk( cudaPeekAtLastError() );
}

#endif /* CHECKSUM_H_ */
