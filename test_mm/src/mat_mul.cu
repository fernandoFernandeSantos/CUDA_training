#include <stdio.h>
#include <stdlib.h>

#include "checksum.h"

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







void print_mat_row_major(double *mat, long m, long n, const char *mat_name) {
	printf("ROW-MAJOR ORDER: printing %s lin %ld col %ld\n", mat_name, m, n);
	long i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++)
			printf("%ld ", (PRINT_TYPE) mat[i * n + j]);
		printf("\n");
	}
	printf("on vector 1d\n");
	for (i = 0; i < m * n; i++) {
		printf("%ld ", (PRINT_TYPE) mat[i]);
	}
	printf("\n");
}

void print_mat_collum_major(double *mat, long m, long n, const char *mat_name) {
	printf("COLLUM-MAJOR ORDER: printing %s lin %ld col %ld\n", mat_name, m, n);
	long i, j;
	for (i = 0; i < m; i++) {

		for (j = 0; j < n; j++) {
			printf("%ld ", (PRINT_TYPE) mat[j * m + i]);
		}
		printf("\n");
	}
	printf("on vector 1d\n");
	for (i = 0; i < m * n; i++) {
		printf("%ld ", (PRINT_TYPE) mat[i]);
	}
	printf("\n");

}

void fill_mat(double* t, long n) {
	long i;
	for (i = 0; i < n; i++) {
		t[i] = 1;
	}
}

void fill_mat_row_major(double *t, long m, long n) {
	long i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			t[i * n + j] = double(i);
}

void fill_mat_collum_major(double *t, long m, long n) {
	long i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			t[j * m + i] = double(i);
}
void compare(double *t, double *s, long siz) {
	long i;
	for (i = 0; i < siz; i++) {
		if (fabs(t[i]) - fabs(s[i]) > 0.0000001)
			printf("t[%ld] is diff from s[%ld] on diff %lf", i, i,
					fabs(t[i]) - fabs(s[i]));
	}
}

cublasStatus_t dgemm_host(int width_a, int height_a, int width_b, int height_b,
		double *a, double *b, double *c) {
	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alpha = 1;
	const double beta = 0;
	//note cublas is column primary!
	//need to transpose the order
	//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA,
	//matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

	cublasStatus_t ret = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width_b,
			height_a, width_a, &alpha, b, width_b, a, width_a, &beta, c,
			width_b);

	if(CUBLAS_STATUS_SUCCESS != ret){
		printf("pau no blas\n");
		exit(-1);
	}

	cublasDestroy(handle);
	return ret;
}

void matrix_multiplication_abft() {
	long lin_a = 7;
	long col_a = 9;
	long lin_b = col_a;
	long col_b = 5;
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
	fill_mat_row_major(host_array_a, lin_a + 1, col_a + 1);
	fill_mat_row_major(host_array_b, lin_b + 1, col_b + 1);

	//cuda memories
	double *device_array_a, *device_array_b, *device_array_c;
	cudaMalloc(&device_array_a, siz_a);
	cudaMalloc(&device_array_b, siz_b);
	cudaMalloc(&device_array_c, siz_c);
	//copy to devicex_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
	cudaMemcpy(device_array_a, host_array_a, siz_a, cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_b, host_array_b, siz_b, cudaMemcpyHostToDevice);



//
//	printf("blocks_abft_first %ld threads_abft_firs %ld\n", blocks_abft_first,
//			threads_abft_first);
//	printf("blocks_abft_second %ld threads_abft_second %ld\n",
//			blocks_abft_second, threads_abft_second);
	first_abraham(device_array_a, lin_a + 1, col_a + 1);
	second_abraham(device_array_b, lin_b + 1, col_b + 1);

	cudaMemcpy(host_array_a, device_array_a, siz_a, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array_b, device_array_b, siz_b, cudaMemcpyDeviceToHost);
	print_mat_row_major(host_array_a, lin_a + 1, col_a + 1, "matrix A");
	printf("\n");
	print_mat_row_major(host_array_b, lin_b + 1, col_b + 1, "matrix B");

	//cublasStatus_t dgemm_host(int width_a, int height_a, int width_b, int height_b, double *a, double *b,	double *c)
	dgemm_host(col_a + 1, lin_a + 1, col_b + 1, lin_b + 1, device_array_a,
			device_array_b, device_array_c);

	cudaMemcpy(host_array_c, device_array_c, siz_c, cudaMemcpyDeviceToHost);
	print_mat_row_major(host_array_c, lin_a + 1, col_b + 1, "GPU result mat");
	int row_detected_errors_host, col_detected_errors_host;

	//abraham_check(device_array_c, (lin_a + 1), (col_b + 1));

	//cudaMemcpyFromSymbol(&row_detected_errors_host, row_detected_errors,sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpyFromSymbol(&col_detected_errors_host, col_detected_errors,sizeof(int), cudaMemcpyDeviceToHost);
	printf("Detected row errors: %d\nDetected collum errors %d\n", row_detected_errors_host, col_detected_errors_host);

	//printf("compare matrices\n");

//	free(host_array_a);
//	free(host_array_b);
//	free(host_array_c);
//	free(host_array_c_temp);
//
//	cudaFree(device_array_a);
//	cudaFree(device_array_b);
//	cudaFree(device_array_c);
}

int main(void) {
	matrix_multiplication_abft();
	return 0;
}
//
//__global__ void mat_cpy(double *dst, double *src, long collums, long rows) {
//	long x = (blockDim.x * blockIdx.x) + threadIdx.x;
//	long y = (blockDim.y * blockIdx.y) + threadIdx.y;
//
//	long index = (collums * y) + x;
//
//	if (collums * rows > index)
//		dst[index] = src[index];
//}
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
//
//__global__ void mat_mult(double *dst, double *a, double *b, long col) {
//	long i = blockIdx.y * blockDim.y + threadIdx.y;
//	long j = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (i > col || j > col)
//		return;
//
//	double acc = 0;
//	long index_dst = i * col + j;
//	long k;
//	for (k = 0; k < col; k++) {
//		acc += a[i * col + k] * b[k * col + j];
//	}
//	dst[index_dst] = acc;
//}
