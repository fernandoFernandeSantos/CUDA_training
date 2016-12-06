//extern "C" {
#include "abft.h"
#include <stdio.h>
//}

//extern "C" {

int use_abft = 0;


void set_use_abft(int n) {
	use_abft = n;
}

int get_use_abft() {
	return use_abft;
}
//}

__device__ ErrorReturn err_count;

__device__ inline long get_index(long i, long j, long n){
	return i * n + j;
}


/* Finds the sum of all elements in the row excluding the element at eRow and the checksum element */
__global__ void excl_row_sum(float *mat, long rows, long cols, long error_row, long error_col) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (mat == NULL || error_row >= rows || error_col >= cols)
    	return;
    for (i = 0; i < cols - 1; i++) {
        /* if i is not the trouble column */
        if (i != cols){
        	long index = get_index(i, error_row, cols);
        	sum += mat[index];
        }
    }
    return;
}

/* Finds the sum of all elements in the col excluding the element at eRow and the checksum element */
__global__ void excl_col_sum(float *mat, long rows, long cols, long error_row) {
    long j = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (mat == NULL || error_row > rows)
    	atomicAdd(&err_count.error_status, 1);
    long i;
    for (i = 0; i < rows - 1; i++) {
        /* if j is not the trouble row */
        if (i != error_row){
        	long index = get_index(i, j, cols);
        	sum += mat[index];
        }
    }
    return;
}



__global__ void check_col(float *mat, long rows, long cols) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	long b_index = i * cols + cols - 1;
	if (cols == 1 || b_index > (rows * cols))
		return;

	long k;
	double acc = 0;
	//must be less one
	for (k = 0; k < cols - 1; k++) {
		acc += (mat[i * cols + k] / DIV_VALUE);
	}

	//printf("b_index %ld acc %lf \n", b_index, acc);
	float diff = fabs(mat[b_index] - acc);
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&err_count.col_detected_errors[0], 1);
//		printf("passou no col mat[%ld] = %ld diff %ld calc %ld i %ld\n",
//				b_index, (long) mat[b_index], (long) diff, (long) acc, i);
	}
	//__syncthreads();
}

__global__ void check_row(float *mat, long rows, long cols) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;
	long a_index = (rows - 1) * cols + j;
	if (rows == 1 || a_index > (rows * cols))
		return;

	long k;
	double acc = 0;
	//must be less one
	for (k = 0; k < rows - 1; k++) {
		acc += (mat[k * cols + j] / DIV_VALUE);
	}
	//printf("a_index %ld acc %lf \n", rows_a * cols_a + j, acc);

	float diff = fabs(mat[a_index] - acc);
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&err_count.row_detected_errors[0], 1);
//		printf("passou no row mat[%ld] = %lf diff %lf calc %lf i value %ld\n",
//				a_index, mat[a_index - 1], diff, acc, j);
	}
	//__syncthreads();
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
__global__ void first_abraham_op(float *a, long rows_a, long cols_a) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;
	long a_index = get_index((rows_a - 1), j, cols_a);

	if (rows_a == 1 || a_index > (rows_a * cols_a))
		return;

	long k;
	double acc = 0;
	for (k = 0; k < rows_a - 1; k++) {
		long index = get_index(k, j, cols_a);
		acc += (a[index] / DIV_VALUE);
	}

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
__global__ void second_abraham_op(float *b, long rows_b, long cols_b) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	long b_index = get_index(i, cols_b - 1, cols_b);
	if (rows_b == 1 || b_index > (rows_b * cols_b))
		return;

	long k;
	double acc = 0;
	for (k = 0; k < cols_b - 1; k++) {
		long index = get_index(i, k, cols_b);
		acc += (b[index] / DIV_VALUE);
	}
	b[b_index] = acc;
}

ErrorReturn check_checksums_from_host(float *c, long rows_c, long cols_c) {
	long blocks = ceil(float(cols_c) / float(BLOCK_SIZE));
	long threads = ceil(float(cols_c) / float(blocks));
	check_row<<<blocks, threads>>>(c, rows_c, cols_c);
	blocks = ceil(float(rows_c) / float(BLOCK_SIZE));
	threads = ceil(float(rows_c) / float(blocks));
	check_col<<<blocks, threads>>>(c, rows_c, cols_c);
}


void calc_checksums_from_host(float *a, float *b, long rows_a, long cols_a, long rows_b, long cols_b) {
	//1d grid for abft operations
//	long *temp;
//	long temp_host[cols_a];
//	cudaMalloc(&temp, cols_a * sizeof(long));

	long blocks = ceil(float(cols_a) / float(BLOCK_SIZE));
	long threads = ceil(float(cols_a) / float(blocks));

	first_abraham_op<<<blocks, threads>>>(a, rows_a, cols_a);

//	cudaMemcpy(temp_host, temp, cols_a * sizeof(long), cudaMemcpyDeviceToHost);


	printf("first blocks %ld threads %ld\n", blocks, threads);
	//second
	blocks = ceil(float(rows_b) / float(BLOCK_SIZE));
	threads = ceil(float(rows_b) / float(blocks));
	second_abraham_op<<<blocks, threads>>>(b, rows_b, cols_b);
	printf("second blocks %ld threads %ld\n", blocks, threads);
	long row_detected_errors_host, col_detected_errors_host;

	cudaMemcpyFromSymbol(&row_detected_errors_host, err_count.row_detected_errors,
			sizeof(int));
	cudaMemcpyFromSymbol(&col_detected_errors_host, err_count.col_detected_errors,
			sizeof(int));
}


__global__ void fault_injection(float *mat, int pos){
	mat[pos] = (pos * 5000);
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
