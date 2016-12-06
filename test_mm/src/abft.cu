//extern "C" {
#include "abft.h"
#include <stdio.h>
//}

//extern "C" {

int use_abft = 0;
__device__ int row_detected_errors = 0;
__device__ int col_detected_errors = 0;

ErrorReturn new_error_return(long rows, long cols) {
	ErrorReturn ret;
	ret.siz_r = sizeof(long);
	ret.siz_c = sizeof(long);
	ret.byte_siz_r = ret.siz_r * rows;
	ret.byte_siz_c = ret.siz_c * cols;
	ret.cols = cols;
	ret.rows = rows;
	//host side
	ret.col_detected_errors_host = (long*) calloc(cols, ret.siz_c);
	ret.row_detected_errors_host = (long*) calloc(rows, ret.siz_r);
	if (ret.col_detected_errors_host == NULL
			|| ret.row_detected_errors_host == NULL) {
		exit(-1);
	}

	//device side
	cudaMalloc(&ret.col_detected_errors_gpu, ret.byte_siz_c);
	cudaMalloc(&ret.row_detected_errors_gpu, ret.byte_siz_r);
	cudaMalloc(&ret.col_err_gpu, ret.byte_siz_c);
	cudaMalloc(&ret.row_err_gpu, ret.byte_siz_r);
	gpuErrchk(cudaDeviceSynchronize());

	ret.error_status = 0;
	return ret;
}

void free_error_return(ErrorReturn *t) {
	free(t->col_detected_errors_host);
	free(t->row_detected_errors_host);
	cudaFree(t->col_detected_errors_gpu);
	cudaFree(t->row_detected_errors_gpu);
	cudaFree(t->row_err_gpu);
	cudaFree(t->col_err_gpu);
}

void cpy_from_device(ErrorReturn *e){

	cudaMemcpy(e->col_detected_errors_host, e->col_detected_errors_gpu, e->byte_siz_c, cudaMemcpyDeviceToHost);
	gpuErrchk(cudaDeviceSynchronize());
	cudaMemcpy(e->row_detected_errors_host, e->row_detected_errors_gpu, e->byte_siz_r, cudaMemcpyDeviceToHost);
	gpuErrchk(cudaDeviceSynchronize());


	cudaMemcpyFromSymbol(&e->row_detected_errors, row_detected_errors,
			sizeof(int));
	cudaMemcpyFromSymbol(&e->col_detected_errors,col_detected_errors,
			sizeof(int));
}

void set_use_abft(int n) {
	use_abft = n;
}

int get_use_abft() {
	return use_abft;
}
//}

__device__ ErrorReturn err_count;

__device__ inline long get_index(long i, long j, long n) {
	return i * n + j;
}

/* Finds the sum of all elements in the row excluding the element at eRow and the checksum element */
__global__ void excl_row_sum(float *mat, long rows, long cols, float *sum,
		long err_row) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	*sum = 0;
	if (mat == NULL || err_row >= rows)
		return;
	long k;
	for (k = 0; k < cols - 1; k++) {
		/* if i is not the trouble column */
		if (i != err_row) {
			long index = get_index(i, k, cols);
			(*sum) += mat[index];
		}
	}
}

/* Finds the sum of all elements in the col excluding the element at eRow and the checksum element */
__global__ void excl_col_sum(float *mat, long rows, long cols, float *sum,
		long err_col) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;
	*sum = 0;
	if (mat == NULL || err_col > cols)
		return;
	long k;
	for (k = 0; k < rows - 1; k++) {
		/* if j is not the trouble row */
		if (j != err_col) {
			long index = get_index(k, j, cols);
			(*sum) += mat[index];
		}
	}
}

__global__ void check_col(float *mat, long rows, long cols, long *col_detected_errors_gpu) {
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
		atomicAdd(&col_detected_errors, 1);
		col_detected_errors_gpu[i] = i;
	}
	//__syncthreads();
}

__global__ void check_row(float *mat, long rows, long cols,long *row_detected_errors_gpu) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;
	long a_index = (rows - 1) * cols + j;
	if (rows == 1 || a_index > (rows * cols))
		return;

	long k;
	double acc = 0;
	//must be less one
	for (k = 0; k < rows - 1; k++) {
		long index = get_index(k, j, cols);
		acc += (mat[index] / DIV_VALUE);
	}
	//printf("a_index %ld acc %lf \n", rows_a * cols_a + j, acc);

	float diff = fabs(mat[a_index] - acc);
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&row_detected_errors, 1);
		row_detected_errors_gpu[j] = j;
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
 acc += b[i * (col_b + 1) + j];.col_err_gpu
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

void print_row_detected_errors(long rows_c, const ErrorReturn& ret) {
	int t;
	printf("Row detected errors\n");
	for (t = 0; t < rows_c; t++) {
		printf("%ld ", ret.row_detected_errors_host[t]);
	}
	printf("\n");
}

__global__ void fault_injection(float *mat, int pos) {
	mat[pos] = (pos * 5000);
}

__global__ void fault_injection_collum(float *mat, int col, int i) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;
	mat[i * col + j] = i * j;

}

ErrorReturn check_checksums_from_host(float *c, long rows_c, long cols_c) {
	ErrorReturn ret = new_error_return(rows_c, cols_c);
	long blocks = ceil(float(cols_c) / float(BLOCK_SIZE));
	long threads = ceil(float(cols_c) / float(blocks));

	int i;
	for(i = 0; i < rows_c - 2; i++){
		fault_injection_collum<<<blocks, threads>>>(c, cols_c, i);
	}

	check_row<<<blocks, threads>>>(c, rows_c, cols_c, ret.row_detected_errors_gpu);
	gpuErrchk(cudaDeviceSynchronize());
	blocks = ceil(float(rows_c) / float(BLOCK_SIZE));
	threads = ceil(float(rows_c) / float(blocks));
	check_col<<<blocks, threads>>>(c, rows_c, cols_c, ret.col_detected_errors_gpu);
	gpuErrchk(cudaDeviceSynchronize());

	cpy_from_device(&ret);
	print_row_detected_errors(rows_c, ret);
	return ret;
}

void calc_checksums_from_host(float *a, float *b, long rows_a, long cols_a,
		long rows_b, long cols_b) {
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
//	long row_detected_errors_host, col_detected_errors_host;
//
//	cudaMemcpyFromSymbol(&row_detected_errors_host, err_count.row_detected_errors,
//			sizeof(int));
//	cudaMemcpyFromSymbol(&col_detected_errors_host, err_count.col_detected_errors,
//			sizeof(int));
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
