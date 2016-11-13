#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include <cublas_v2.h>

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
//rows_b MUST BE THE SAME OF cols_a
__global__ void first_abraham_op(double *a, long rows_a, long cols_a_rows_b) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;
	long i = blockIdx.y * blockDim.y + threadIdx.y;

	//it is so much work for a small task, but in this way i can do everything in a O(row_a) complexity
	//first I calculate the checksum values
	if(((i + 1) % rows_a == 0) && (i > 0)){
		//iterate on j dimension
		long k;
		double acc = 0;
		for (k = 0; k < rows_a; k++) {
			acc += a[j * rows_a + k];
		}
		printf("acc %lf on pos %d\n", acc, rows_a * j + rows_a);
//		printf("passou dentro acc %lf rows_a * cols_a_rows_b + j %ld\n",acc, (rows_a) * (cols_a_rows_b - 1) + j);

		a[(rows_a) * j + rows_a - 1] = acc;
	}
	//so when I could add a extra line and collum, there will be a blanck collum for matrix A
//	if(((j + 1) % cols_a_rows_b == 0) && (j > 0)){
//		a[i * cols_a_rows_b + cols_a_rows_b - 1] = 0;
//
//	}

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
__global__ void second_abraham_op(double *b, long rows_b_cols_a, long collums_b) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;
	long i = blockIdx.y * blockDim.y + threadIdx.y;

	//printf("j %ld rows_b %ld j mod rows %ld\n", j, rows_b_cols_a, j % rows_b_cols_a);
	if(((j + 1) % rows_b_cols_a == 0) && (j > 0)){
		long k;
		double acc = 0;
		for (k = 0; k < collums_b; k++) {
			acc += b[j * collums_b + k];
		}
		//printf("dentro acc %lf j * collums_b + collums_b %ld\n", acc, j * collums_b + collums_b);
		b[(collums_b - 1) * collums_b + i] = acc;
	}

//	if(((i + 1) % rows_b_cols_a == 0) && (i > 0)){
//		b[(rows_b_cols_a - 1) * collums_b + j] = 0;
//	}
}

__global__ void mat_mult(double *dst, double *a, double *b, long col) {
	long i = blockIdx.y * blockDim.y + threadIdx.y;
	long j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > col || j > col)
		return;

	double acc = 0;
	long index_dst = i * col + j;
	long k;
	for (k = 0; k < col; k++) {
		acc += a[i * col + k] * b[k * col + j];
	}
	dst[index_dst] = acc;
}

void print_mat_row_major(double *mat, long m, long n, const char *mat_name) {
	printf("ROW-MAJOR ORDER: printing %s lin %ld col %ld\n", mat_name, m, n);
	long i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++)
			printf("%ld ", (PRINT_TYPE) mat[i * n + j]);
		printf("\n");
	}
}

void print_mat_collum_major(double *mat, long m, long n, const char *mat_name){
	printf("COLLUM-MAJOR ORDER: printing %s lin %ld col %ld\n", mat_name, m, n);
	long i, j;
	for(i = 0; i < m; i++){

		for(j = 0; j < n; j++){
			printf("%ld ",(PRINT_TYPE)mat[j*m + i]);
		}
		printf("\n");
	}
//	printf("on vector 1d\n");
//	for(i = 0; i < m * n; i++){
//		printf("%ld ", (PRINT_TYPE)mat[i]);
//	}
//	printf("\n");

}

void fill_mat(double* t, long n) {
	long i;
	for (i = 0; i < n; i++) {
		t[i] = 1;
	}
}


void fill_mat_mn(double *t, long m, long n){
	long i,j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			if(i == j)
				t[j*m + i] = double(i);
}
void compare(double *t, double *s, long siz) {
	long i;
	for (i = 0; i < siz; i++) {
		if (fabs(t[i]) - fabs(s[i]) > 0.0000001)
			printf("t[%ld] is diff from s[%ld] on diff %lf", i, i,
					fabs(t[i]) - fabs(s[i]));
	}
}


//emm(cublasHandle_t handle,
//                           cublasOperation_t transa, cublasOperation_t transb,
//                           int m, int n, int k,
//                           const double          *alpha,
//                           const double          *A, int lda,
//                           const double          *B, int ldb,
//                           const double          *beta,
//                           double          *C, int ldc)
//
//
//Read more at: http://docs.nvidia.com/cuda/cublas/index.html#ixzz4PlFurrom
//Follow us: @GPUComputing on Twitter | NVIDIA on Facebook

cublasStatus_t dgemm_host(int m, int n, int k, double *a, double *b, double *c) {
	cublasHandle_t handle;
	cublasCreate(&handle);
	int lda = m, ldb = k, ldc = m;
	 const double alf = 1;
	 const double bet = 0;
	 const double *alpha = &alf;
	 const double *beta = &bet;
	cublasStatus_t ret = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha,
			a, lda, b, ldb, beta, c, ldc);

	cublasDestroy(handle);
	return ret;
}

void matrix_multiplication_abft() {
	long size = 10;
	long lin_a = 10;
	long col_a = 20;
	long lin_b = col_a;
	long col_b = 15;
	long vec_siz_a = ((lin_a + 1) * (col_a + 1));
	long vec_siz_b = ((lin_b + 1) * (col_b + 1));
	long vec_siz_c = ((lin_a + 1) * (col_b + 1));
	const long siz_a = vec_siz_a * sizeof(double);
	const long siz_b = vec_siz_b * sizeof(double);
	const long siz_c = vec_siz_b * sizeof(double);
	//host memories
	double* host_array_a = (double*) calloc(vec_siz_a, sizeof(double));
	double* host_array_b = (double*) calloc(vec_siz_b, sizeof(double));
	double* host_array_c = (double*) calloc(vec_siz_c, sizeof(double));
	double* host_array_c_temp = (double*) calloc(vec_siz_c, sizeof(double));
	fill_mat_mn(host_array_a, lin_a + 1, col_a + 1);
	fill_mat_mn(host_array_b, lin_b + 1, col_b + 1);

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

	//2d grid for abft operations

	long blocks_abft_first = ceil((lin_a + 1) / float(BLOCK_SIZE));
	long threads_abft_first = ceil((col_a + 1) / float(blocks_abft_first));
	dim3 gridDimABFT_1st(blocks_abft_first, blocks_abft_first);
	dim3 blockDimABFT_1st(threads_abft_first, threads_abft_first);

	//second
	long blocks_abft_second = ceil((col_b + 1) / float(BLOCK_SIZE));
	long threads_abft_second = ceil((lin_b + 1) / float(blocks_abft_second));
	dim3 gridDimABFT_2nd(blocks_abft_second, blocks_abft_second);
	dim3 blockDimABFT_2nd(threads_abft_second, threads_abft_second);

	printf("blocks_abft_first %ld threads_abft_firs %ld\n", blocks_abft_first, threads_abft_first);
	printf("blocks_abft_second %ld threads_abft_second %ld\n", blocks_abft_second, threads_abft_second);
	first_abraham_op<<<gridDimABFT_1st, blockDimABFT_1st>>>(device_array_a, lin_a + 1,
			col_a + 1);
//	second_abraham_op<<<gridDimABFT_2nd, blockDimABFT_2nd>>>(device_array_b, lin_b + 1,
//			col_b + 1);

	cudaMemcpy(host_array_a, device_array_a, siz_a, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array_b, device_array_b, siz_b, cudaMemcpyDeviceToHost);
	print_mat_collum_major(host_array_a, lin_a + 1, col_a + 1, "matrix A");
	printf("\n");
	print_mat_collum_major(host_array_b, lin_b + 1, col_b + 1, "matrix B");
//	mat_mult<<<gridDim, blockDim>>>(device_array_c, device_array_a,
//			device_array_b, col_b);
//	dgemm_host(lin_a + 1,col_b + 1,col_a + 1, device_array_a, device_array_b, device_array_c);
	printf("\nblocks %ld threads %ld\n", blocks, threads);
	cudaMemcpy(host_array_c, device_array_c, siz_c, cudaMemcpyDeviceToHost);
	print_mat_row_major(host_array_c, lin_a + 1, col_b + 1, "GPU result mat");
	printf("compare matrices\n");
	//compare(host_array_c, host_array_c_temp, VECTOR_SIZE_C);
	cudaFree(device_array_a);
	printf("passou a\n");
	cudaFree(device_array_b);
	printf("passou array b\n");
	cudaFree(device_array_c);
	printf("problem with host arrays\n");
	free(host_array_a);
	free(host_array_b);
	free(host_array_c);
	free(host_array_c_temp);
}

int main(void) {
	matrix_multiplication_abft();
	return 0;
}
//
//int gemm_ongpu_abft(double *a, double *b, double *c, long lin_a, long col_a,
//		long lin_b, long col_b) {
//	long i, j;
//	double acc = 0;
//	int ret = 0;
//	long col_c = col_b;
////	long lin_c = lin_a;
//	//first ABRAHAM operation
//	for (j = 0; j < col_a; j++) {
//		acc = 0;
//		for (i = 0; i < lin_a; i++)
//
//			acc += a[i * col_a + j];
//
//		a[lin_a * col_a + j] = acc;
//	}
//
//	//second ABRAHAM operation
//	for (i = 0; i < lin_b; i++) {
//		acc = 0;
//		for (j = 0; j < col_b; j++)
//			acc += b[i * (col_b + 1) + j];
//		//printf("i * col_b %ld col b %ld  acc %lf\n", i * col_b, col_b, acc);
//		b[i * (col_b + 1) + col_b] = acc;
//	}
//
//	//print_mat(a, lin_a + 1, col_a);
//	//printf("\n");
//	//print_mat(b, lin_b, col_b + 1);
//	//performs matrix multiplication
//	gemm_1d(a, b, c, lin_a + 1, col_a, lin_b, col_b + 1, col_b + 1, lin_a + 1);
//
//	//check all checksums
//	//line checksum
//	for (j = 0; j < col_a; j++) {
//		acc = 0;
//		for (i = 0; i < lin_a; i++)
//			acc += c[i * col_c + j];
//
//		if (fabs(c[lin_a * col_c + j]) - fabs(acc) >= MAX_THRESHOLD) {
////			printf(
////					"lin - position corrupted [%ld][%ld] - exp chsum %lf got chsum %lf diff - %lf\n",
////					lin_a, j, c[lin_a * col_c + j], acc,
////					c[lin_a * col_c + j] - acc);
//			ret++;
//		}
//	}
//
//	//collum checksum
//	for (i = 0; i < lin_b; i++) {
//		acc = 0;
//		for (j = 0; j < col_b; j++)
//			acc += c[i * col_c + j];
//
//		if (fabs(c[i * col_c + col_b] - acc) >= MAX_THRESHOLD) {
////			printf(
////					"collum - position corrupted [%ld][%ld] - exp chsum %lf got chsum %lf diff %lf\n",
////					i, col_b, c[i * col_c + col_b], acc,
////					c[i * col_c + col_b] - acc);
//			ret++;
//		}
//	}
//	return ret;
//
//}
//
//void matrix_multiplication_no_abft() {
//	const long siz_a = VECTOR_SIZE_A * sizeof(double);
//	const long siz_b = VECTOR_SIZE_B * sizeof(double);
//	const long siz_c = VECTOR_SIZE_C * sizeof(double);
//	//host memories
//	double* host_array_a = (double*) calloc(VECTOR_SIZE_A, sizeof(double));
//	double* host_array_b = (double*) calloc(VECTOR_SIZE_B, sizeof(double));
//	double* host_array_c = (double*) calloc(VECTOR_SIZE_C, sizeof(double));
//	double* host_array_c_temp = (double*) calloc(VECTOR_SIZE_C, sizeof(double));
//	fill_mat(host_array_a, VECTOR_SIZE_A);
//	fill_mat(host_array_b, VECTOR_SIZE_B);
//	//print_mat(host_array_a, COLLUMS_A, ROWS_A, "matrix A");
//	printf("\n");
//	//print_mat(host_array_b, COLLUMS_B, ROWS_B, "matrix B");
//	//perform host matrix multiplication
//	//	gemm_1d(host_array_a, host_array_b, host_array_c_temp, ROWS_A, COLLUMS_A,
//	//			ROWS_B, COLLUMS_B, ROWS_A, COLLUMS_B);
//	//print_mat(host_array_c_temp, COLLUMS_B, ROWS_A, "matrix C temp");
//	//cuda memories
//	double *device_array_a, *device_array_b, *device_array_c;
//	cudaMalloc(&device_array_a, siz_a);
//	cudaMalloc(&device_array_b, siz_b);
//	cudaMalloc(&device_array_c, siz_c);
//	//copy to device
//	cudaMemcpy(device_array_a, host_array_a, siz_a, cudaMemcpyHostToDevice);
//	cudaMemcpy(device_array_b, host_array_b, siz_b, cudaMemcpyHostToDevice);
//	//kernel parameters
//	//we know that each block has 1024 threads
//	long blocks = ceil(N / float(BLOCK_SIZE));
//	long threads = ceil(N / float(blocks));
//	//2d grid
//	dim3 gridDim(blocks, blocks);
//	//threads num, 2d
//	dim3 blockDim(threads, threads);
//	mat_mult<<<gridDim, blockDim>>>(device_array_c, device_array_a,
//			device_array_b, N);
//	printf("\nblocks %ld threads %ld\n", blocks, threads);
//	cudaMemcpy(host_array_c, device_array_c, siz_c, cudaMemcpyDeviceToHost);
//	//print_mat(host_array_c, COLLUMS_A, ROWS_A, "GPU result mat");
//	printf("compare matrices\n");
//	//compare(host_array_c, host_array_c_temp, VECTOR_SIZE_C);
//	cudaFree(device_array_a);
//	cudaFree(device_array_b);
//	cudaFree(device_array_c);
//	free(host_array_a);
//	free(host_array_b);
//	free(host_array_c);
//	free(host_array_c_temp);
//}
