//extern "C" {
#include "abft.h"
#include <stdio.h>
//}

//extern "C" {

int use_abft = 0;
__device__ int row_detected_errors = 0;
__device__ int col_detected_errors = 0;

void print_row_detected_errors(const ErrorReturn& ret) {
	int t;
	printf("Row detected errors rows_c %ld\n", ret.rows);
	for (t = 0; t < ret.rows; t++) {
		printf("%ld ", ret.row_detected_errors_host[t]);
	}
	printf("\n");
}

void print_col_detected_errors(const ErrorReturn& ret) {
	int t;
	printf("Col detected errors cols_c %ld\n", ret.cols);
	for (t = 0; t < ret.cols; t++) {
		printf("%ld ", ret.col_detected_errors_host[t]);
	}
	printf("\n");
}

ErrorReturn new_error_return(long_t rows, long_t cols) {
	ErrorReturn ret;
	ret.siz_r = sizeof(long_t);
	ret.siz_c = sizeof(long_t);
	ret.byte_siz_r = ret.siz_r * rows;
	ret.byte_siz_c = ret.siz_c * cols;
	ret.cols = cols;
	ret.rows = rows;
	//host side
	ret.col_detected_errors_host = (long_t*) calloc(cols, ret.siz_c);
	ret.row_detected_errors_host = (long_t*) calloc(rows, ret.siz_r);
	if (ret.col_detected_errors_host == NULL
			|| ret.row_detected_errors_host == NULL) {
		exit(-1);
	}

	//device side
	cudaMalloc(&ret.col_detected_errors_gpu, ret.byte_siz_c);
	//cudaMemset(&ret.col_detected_errors_gpu, 0, ret.byte_siz_c);
	cudaMalloc(&ret.row_detected_errors_gpu, ret.byte_siz_r);
	//cudaMemset(&ret.row_detected_errors_gpu, 0, ret.byte_siz_r);
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

void cpy_from_device(ErrorReturn *e) {
	//print_col_detected_errors(*e);
//	printf("byte_siz-c %ld\n", e->byte_siz_c);
	cudaMemcpy(e->col_detected_errors_host, e->col_detected_errors_gpu,
			e->byte_siz_c, cudaMemcpyDeviceToHost);
	//print_col_detected_errors(*e);
	//gpuErrchk(cudaDeviceSynchronize());
	cudaMemcpy(e->row_detected_errors_host, e->row_detected_errors_gpu,
			e->byte_siz_r, cudaMemcpyDeviceToHost);
	//gpuErrchk(cudaDeviceSynchronize());

	cudaMemcpyFromSymbol(&e->row_detected_errors, row_detected_errors,
			sizeof(int));
	cudaMemcpyFromSymbol(&e->col_detected_errors, col_detected_errors,
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

__device__  inline long_t get_index(long_t i, long_t j, long_t n) {
	return i * n + j;
}

/**
 * check all collums if they are right
 * input
 * mat resulting matrix
 * rows mat rows
 * cols mat cols
 * error vector
 * check all colluns against checksum
 * NUM OF THREADS MUST BE THE SAME AS COLLUMS
 * */
__global__ void check_cols(float *mat, long_t rows, long_t cols,
		long_t *col_detected_errors_gpu) {
	long_t j = blockIdx.x * blockDim.x + threadIdx.x;
	col_detected_errors_gpu[j] = 0;
	long_t mat_index = get_index((rows - 1), j, cols);

	if (rows == 1 || mat_index > (rows * cols))
		mat[0] = 0;

	long_t k;
	double acc = 0;
	for (k = 0; k < rows - 1; k++) {
		long_t index = get_index(k, j, cols);
		acc += (mat[index] / DIV_VALUE);
	}

//	mat[mat_index] = acc;
	//printf("b_index %ld acc %lf \n", b_index, acc);
	float diff = fabs(mat[mat_index] - acc);
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&col_detected_errors, 1);
		col_detected_errors_gpu[j] = j;
		//printf("foi %ld\n", j);
	} else {
		col_detected_errors_gpu[j] = -1;
	}

	//__syncthreads();
}

/*check all rows against checksum*/
__global__ void check_rows(float *mat, long_t rows, long_t cols,
		long_t *row_detected_errors_gpu) {
	long_t i = blockIdx.x * blockDim.x + threadIdx.x;

	long_t mat_index = get_index(i, cols - 1, cols);
	if (rows == 1 || mat_index > (rows * cols))
		return;

	long_t k;
	double acc = 0;
	for (k = 0; k < cols - 1; k++) {
		long_t index = get_index(i, k, cols);
		acc += (mat[index] / DIV_VALUE);
	}

	//printf("a_index %ld acc %lf \n", rows_a * cols_a + i, acc);

	float diff = fabs(mat[mat_index] - acc);
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&row_detected_errors, 1);
		printf("\nthread %ld\n", i);
		row_detected_errors_gpu[i] = i;
//		printf("passou no row mat[%ld] = %lf diff %lf calc %lf i value %ld\n",
//				a_index, mat[a_index - 1], diff, acc, j);
	} else {
		row_detected_errors_gpu[i] = -1;
	}
	//__syncthreads();
}

//**********************************************************************************

/* Finds the sum of all elements in the row excluding the element at eRow and the checksum element */
__global__ void excl_row_sum(float *mat, long_t rows, long_t cols, float *sum,
		long_t err_row) {
	long_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (mat == NULL || err_row >= rows)
		return;
	long_t k;
	for (k = 0; k < cols - 1; k++) {
		/* if i is not the trouble column */
		if (i != err_row) {
			long_t index = get_index(i, k, cols);
			atomicAdd(sum, mat[index]);
		}
	}
}

__device__ float excl_row_sum_seq(float *mat, long_t rows, long_t cols,
		long_t wrong_row) {
	long_t i = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;
	if (mat == NULL || wrong_row >= rows)
		return sum;
	long_t k;
	for (k = 0; k < cols - 1; k++) {
		/* if i is not the trouble column */
		if (i != wrong_row) {
			long_t index = get_index(i, k, cols);
			sum += mat[index];
		}
	}
	return sum;
}
/* Finds the sum of all elements in the col excluding the element at eRow and the checksum element
 * err_col will be calculated here, avoiding memory copy
 * -- the parameters are
 * mat
 * rows
 * cols
 * col_detected_errors_gpu
 * col_detected_errors
 * */

__global__ void excl_col_sum(float *mat, long_t rows, long_t cols, float *sum,
		long_t err_col) {
	long_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if (mat == NULL || err_col > cols)
		return;
	long_t k;
	//for (k = 0; k < rows - 1; k++) {
	/* if j is not the trouble row */
	if (j != err_col) {
		long_t index = get_index(k, j, cols);
		atomicAdd(sum, mat[index]);
	}
	//}
}

__device__ float excl_col_sum_seq(float *mat, long_t rows, long_t cols,
		long_t wrong_col) {
	long_t j = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;
	if (mat == NULL || wrong_col > cols)
		return sum;
	long_t k;
	for (k = 0; k < rows - 1; k++) {
		/* if j is not the trouble row */
		if (j != wrong_col) {
			long_t index = get_index(k, j, cols);
			sum += mat[index];
			//atomicAdd(sum, mat[index]);
		}
	}
	return sum;
}
//###############################################################################

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
__global__ void calc_collum_checksum(float *mat, long_t rows, long_t cols) {
	long_t j = blockIdx.x * blockDim.x + threadIdx.x;
	long_t mat_index = get_index((rows - 1), j, cols);

	if (rows == 1 || mat_index > (rows * cols))
		mat[0] = 0;

	long_t k;
	double acc = 0;
	for (k = 0; k < rows - 1; k++) {
		long_t index = get_index(k, j, cols);
		acc += (mat[index] / DIV_VALUE);
	}

	mat[mat_index] = acc;
}

/**
 * 	for (i = 0; i < lin_b; i++) {
 acc = 0;
 for (j = 0; j < col_b; j++){
 acc += b[i * (col_b + 1) + j];
 }
 b[i * (col_b + 1) + col_b] = acc;

 }
 */
__global__ void calc_row_checksum(float *mat, long_t rows, long_t cols) {
	long_t i = blockIdx.x * blockDim.x + threadIdx.x;
	long_t b_index = get_index(i, cols - 1, cols);
	if (rows == 1 || b_index > (rows * cols))
		return;

	long_t k;
	double acc = 0;
	for (k = 0; k < cols - 1; k++) {
		long_t index = get_index(i, k, cols);
		acc += (mat[index] / DIV_VALUE);
	}
	mat[b_index] = acc;
}

__global__ void fault_injection(float *mat, int pos) {
	mat[pos] = (pos * 5000);
}

__global__ void fault_injection_collum(float *mat, int col, int i) {
	long_t j = blockIdx.x * blockDim.x + threadIdx.x;
	mat[i * col + j] = i * j;

}

//		col = colE[0];
//		if (rowErrors > rows) {
//			errExit("Row errors exceeds rows.");
//		}
//		/* Assumes rowE is in order */
//		for (i = 0; i < rowErrors; i++) {
//			row = rowE[i];
//			sum = exclRowSum(row, col, rows, cols, matrix);
//			/* Not checksum column */
//			if (col != cols - 1) {
//				/* sum of row excluding the current element */
//				matrix[row][col] = matrix[row][cols - 1] - sum;
//			} else {
//				matrix[row][col] = sum;
//			}
//			(*nCorrected)++;
//		}
__global__ void correct_row_device(float *mat, long_t *rows_to_correct,
		long_t *cols_to_correct, long_t rows, long_t cols) {
	long_t i = blockIdx.y * blockDim.y + threadIdx.y;
	long_t j = blockIdx.x * blockDim.x + threadIdx.x;

	long_t row_e = rows_to_correct[i];
	long_t col_e = cols_to_correct[j];

	if (row_e != -1 && col_e != -1) {
		float sum = excl_col_sum_seq(mat, rows, cols, col_e);
		printf("sum %lf i == %ld\n", sum, i);
		long_t index = get_index(row_e, col_e, cols);
		if (col_e != cols - 1) {
			long_t index_e = get_index(row_e, cols - 1, cols);
			mat[index] = mat[index_e] - sum;
		} else {
			mat[index] = sum;
		}
	}
}

//row = rowE[0];
//if (colErrors > cols) {
//    errExit("Column errors exceeds columns.");
//}
///* Assumes colE is in order */
//for (i = 0; i < colErrors; i++) {
//    col = colE[i];
//    /* sum of row excluding the current element */
//    sum = exclColSum(row, col, rows, cols, matrix);
//    if (row != rows - 1) {
//        matrix[row][col] = matrix[rows - 1][col] - sum;
//
//    }
//    else {
//        matrix[row][col] = sum;
//    }
//    (*nCorrected)++;
//}
__global__ void correct_col_device(float *mat, long_t *cols_to_correct, long_t *rows_to_correct, long_t rows, long_t cols) {
	long_t j = blockIdx.x * blockDim.x + threadIdx.x;
	long_t i = blockIdx.y * blockDim.y + threadIdx.y;

	if(i >= rows && j >= cols)
		return;


	long_t col_e = cols_to_correct[j];
	long_t row_e = rows_to_correct[i];
	if (col_e != -1 && row_e != -1) {
		float sum = excl_col_sum_seq(mat, rows, cols, col_e);
		long_t index = get_index(row_e, col_e, cols);

		if (row_e != rows - 1) {
			long index_e = get_index(rows - 1, col_e, cols);
			printf("mat[index] %lf sum %ld\n", mat[index], sum);

			mat[index] = mat[index_e] - sum;
		} else {
			mat[index] = sum;
			printf("passou onde não podia\n");
		}
	}
}

unsigned char correct_host(float *mat, long_t rows, long_t cols,
		ErrorReturn *error) {
	unsigned char ret = 1;
	float *row_sum_vector;
	float *col_sum_vector;
	long_t siz_csv = sizeof(float) * cols;
	long_t siz_rsv = sizeof(float) * rows;

	cudaMalloc(&row_sum_vector, siz_rsv);
	cudaMalloc(&col_sum_vector, siz_csv);
	//iterate in all positions on row_to_correct
	//call everyone, even it's not wrong, it is easier
	long_t blocks_rows = ceil(float(rows) / float(BLOCK_SIZE));
	long_t threads_rows = ceil(float(rows) / float(blocks_rows));
	long_t blocks_cols = ceil(float(rows) / float(BLOCK_SIZE));
	long_t threads_cols = ceil(float(rows) / float(blocks_cols));

	dim3 blocks(blocks_rows, blocks_cols);
	dim3 threads(threads_rows, threads_cols);
	//**************************************************************
	/* Single error */
	printf("\n\npassou aqui row errors %d col errors %d\n\n",
			error->row_detected_errors, error->col_detected_errors);
	if (error->row_detected_errors == 1 && error->col_detected_errors == 1) {

//		correct_row_device<<<1,1>>>(mat, rows_to_correct, )
	} else if (error->row_detected_errors >= 2
			&& error->col_detected_errors == 1) {

		correct_row_device<<<blocks, threads>>>(mat,
				error->row_detected_errors_gpu, error->col_detected_errors_gpu, rows, cols);
		gpuErrchk(cudaDeviceSynchronize());
	} else if (error->col_detected_errors >= 2
			&& error->row_detected_errors == 1) {
		correct_col_device<<<blocks, threads>>>(mat,
				error->col_detected_errors_gpu,
				error->row_detected_errors_gpu, rows, cols);
		gpuErrchk(cudaDeviceSynchronize());
	} else {
		error->could_correct = 0;
	}
	cudaFree(row_sum_vector);
	cudaFree(col_sum_vector);
	return ret;
}

ErrorReturn check_checksums_from_host(float *c, long_t rows_c, long_t cols_c) {
	ErrorReturn ret = new_error_return(rows_c, cols_c);
	long_t blocks_cols = ceil(float(cols_c) / float(BLOCK_SIZE));
	long_t threads_cols = ceil(float(cols_c) / float(blocks_cols));

#ifdef FI
	int i;
//	for(i = 0; i < rows_c - 2; i++){
	fault_injection_collum<<<blocks_cols, threads_cols / 2>>>(c, cols_c, 2);
//	}
#endif
	check_cols<<<blocks_cols, threads_cols>>>(c, rows_c, cols_c,
			ret.col_detected_errors_gpu);
	gpuErrchk(cudaDeviceSynchronize());

	long_t blocks_rows = ceil(float(rows_c) / float(BLOCK_SIZE));
	long_t threads_rows = ceil(float(rows_c) / float(blocks_cols));
	check_rows<<<blocks_rows, threads_rows>>>(c, rows_c, cols_c,
			ret.row_detected_errors_gpu);

	gpuErrchk(cudaDeviceSynchronize());
	cpy_from_device(&ret);

	correct_host(c, rows_c, cols_c, &ret);
	print_row_detected_errors(ret);
	print_col_detected_errors(ret);
	return ret;
}

void calc_checksums_from_host(float *a, float *b, long_t rows_a, long_t cols_a,
		long_t rows_b, long_t cols_b) {
	//1d grid for abft operations
//	long_t *temp;
//	long_t temp_host[cols_a];
//	cudaMalloc(&temp, cols_a * sizeof(long_t));
	long_t blocks = ceil(float(cols_a) / float(BLOCK_SIZE));
	long_t threads = ceil(float(cols_a) / float(blocks));
	calc_collum_checksum<<<blocks, threads>>>(a, rows_a, cols_a);
//	cudaMemcpy(temp_host, temp, cols_a * sizeof(long_t), cudaMemcpyDeviceToHost);
	printf("first blocks %ld threads %ld\n", blocks, threads);
	//second
	blocks = ceil(float(rows_b) / float(BLOCK_SIZE));
	threads = ceil(float(rows_b) / float(blocks));
	calc_row_checksum<<<blocks, threads>>>(b, rows_b, cols_b);
	printf("second blocks %ld threads %ld\n", blocks, threads);
//	long_t row_detected_errors_host, col_detected_errors_host;
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
