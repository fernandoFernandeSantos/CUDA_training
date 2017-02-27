/*
 * abft.h
 *
 *  Created on: 16/11/2016
 *      Author: fernando
 */

#ifndef ABFT_H_
#define ABFT_H_

#include <stdio.h>
#include "cuda_runtime.h"
#include <cublas_v2.h>

#define MAX_THRESHOLD  0.05

#define BLOCK_SIZE 1024

#define DIV_VALUE 1e0 //5

typedef long long_t;
typedef float float_t;

typedef struct erro_return {
	long_t rows;
	long_t cols;
	long_t siz_r;
	long_t siz_c;
	long_t byte_siz_r;
	long_t byte_siz_c;

	long_t* row_detected_errors_host;
	long_t* col_detected_errors_host;

	long_t* row_detected_errors_gpu;
	long_t* col_detected_errors_gpu;


	long_t* col_err_gpu;
	long_t* row_err_gpu;
	long_t* col_err_host;
	long_t* row_err_host;

	int error_status;

	int row_detected_errors;
	int col_detected_errors;
	unsigned char could_correct;
} ErrorReturn;


typedef struct DeviceErrorCounters{
	int row_detected_errors;
	int col_detected_errors;
	float_t sum;

	long_t* row_detected_errors_gpu;
	long_t* col_detected_errors_gpu;
} DeviceErrorCounters;

void calc_checksums_from_host(float_t *a, float_t *b, long_t rows_a, long_t cols_a,long_t rows_b, long_t cols_b);
ErrorReturn check_checksums_from_host(float_t *c, long_t rows_c, long_t cols_c);


void set_use_abft(int n);
int get_use_abft();



inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void free_error_return(ErrorReturn*);
void calc_checksums(float_t *mat_a, float_t *mat_b, float_t *dev_mat,
		float_t *check_row, float_t *check_col, long_t rows_a, long_t cols_a,
		long_t cols_b);


cublasStatus_t dgemm_host(int width_a, int height_a, int width_b, int height_b,
		float *a, float *b, float *c);

cublasStatus_t dgemv_host(int width_a, int height_a, int width_b,
		int height_b, float *a, float *b, float *c, cublasOperation_t trans);

#endif /* ABFT_H_ */
