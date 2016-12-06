/*
 * abft.h
 *
 *  Created on: 16/11/2016
 *      Author: fernando
 */

#ifndef ABFT_H_
#define ABFT_H_

#include <stdio.h>

#define MAX_THRESHOLD  0.05

#define BLOCK_SIZE 1024

#define DIV_VALUE 1e5

typedef struct erro_return {
	int* row_detected_errors;
	int* col_detected_errors;

	int* row_detected_errors_gpu;
	int* col_detected_errors_gpu;

	int error_status;
} ErrorReturn;

void calc_checksums_from_host(float *a, float *b, long rows_a, long cols_a,long rows_b, long cols_b);
ErrorReturn check_checksums_from_host(float *c, long rows_c, long cols_c);


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


#endif /* ABFT_H_ */
