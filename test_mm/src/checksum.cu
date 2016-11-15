#include "checksum.h"

__device__ int row_detected_errors = 0;
__device__ int col_detected_errors = 0;

__device__ void check_col(double *mat, long rows, long cols){
	long i = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	double acc = 0;
	//must be less one
	for (k = 0; k < cols - 1; k++) {
		acc += mat[i * cols + k];
	}
	long b_index = i * cols + cols - 1;
	//printf("b_index %ld acc %lf \n", b_index, acc);

	if(fabs(mat[b_index]) - fabs(acc)){
		atomicAdd(&col_detected_errors, 1);
	}

}



__device__ void check_row(double *mat, long rows, long cols){
	long j = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	double acc = 0;
	//must be less one
	for (k = 0; k < rows - 1; k++) {
		acc += mat[k * cols + j];
	}
	//printf("a_index %ld acc %lf \n", rows_a * cols_a + j, acc);
	long a_index = (rows - 1) * cols + j;
	if(fabs(mat[a_index]) - fabs(acc) <= MAX_THRESHOLD){
		atomicAdd(&row_detected_errors, 1);
	}

}

//DYNAMIC PARALLELISM ONLY TO CALL NEW KERNELS, ARE FUCK KIDDING???
//man, I am so lazy
__global__ void check_checksums(double *c, long rows_c, long cols_c){
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	//rows
	if (i == 0){
		long blocks = ceil(cols_c / BLOCK_SIZE);
		long threads = ceil(cols_c / blocks);
		check_row<<<blocks, threads>>>(c, rows_c, cols_c);
	}
	//cols
	if(i == 1){
		long blocks = ceil(rows_c / BLOCK_SIZE);
		long threads = ceil(rows_c / blocks);
		check_col<<<blocks, threads>>>(c, rows_c, cols_c);
	}
}
