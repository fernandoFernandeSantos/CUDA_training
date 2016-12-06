#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <math.h>
#include "abft.h"
#define PRINT_TYPE double


inline double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void print_mat_row_major(float *mat, long m, long n, const char *mat_name) {
	if (m * n > 5000)
		return;
	printf("ROW-MAJOR ORDER: printing %s lin %ld col %ld\n", mat_name, m, n);
	long i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++)
			printf("%lf ", (PRINT_TYPE) mat[i * n + j]);
		printf("\n");
	}
//	printf("on vector 1d\n");
//	for (i = 0; i < m * n; i++) {
//		printf("%ld ", (PRINT_TYPE) mat[i]);
//	}
	printf("\n");
}

void fill_mat(float* t, long n) {
	long i;
	for (i = 0; i < n; i++) {
		t[i] = 1;
	}
}

void fill_mat_row_major(float *t, long m, long n) {
	long i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			t[i * n + j] = 1; //((rand() % 15) / 3.14578);
}

void compare(float *t, float *s, long siz) {
	long i;
	for (i = 0; i < siz; i++) {
		if (fabs(t[i]) - fabs(s[i]) > 0.0000001)
			printf("t[%ld] is diff from s[%ld] on diff %lf", i, i,
					fabs(t[i]) - fabs(s[i]));
	}
}

/**
 Matrix multiplication: C = A * B.
 Host code.

 This sample implements matrix multiplication as described in Chapter 3
 of the programming guide and uses the CUBLAS library to demonstrate
 the best performance.

 SOME PRECAUTIONS:
 IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
 WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
 The reason is explained as follows:

 CUBLAS library uses column-major storage, but C/C++ use row-major storage.
 When passing the matrix pointer to CUBLAS, the memory layout alters from
 row-major to column-major, which is equivalent to an implicit transpose.

 In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
 C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
 implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
 If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
 multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
 is a column-based cublas matrix, which means C(T) in C/C++, we need extra
 transpose code to convert it to a row-based C/C++ matrix.

 To solve the problem, let's consider our desired result C, a row-major matrix.
 In cublas format, it is C(T) actually (because of the implicit transpose).
 C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
 happen to be C/C++ matrice B and A (still because of the implicit transpose)!
 We don't need extra transpose code, we only need alter the input order!

 CUBLAS provides high-performance matrix multiplication.
 See also:
 V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

cublasStatus_t dgemm_host(int width_a, int height_a, int width_b, int height_b,
		float *a, float *b, float *c) {
	cublasHandle_t handle;
	cublasCreate(&handle);
	const float alpha = 1;
	const float beta = 0;
	//note cublas is column primary!
	//need to transpose the order
//	m input	number of rows of matrix op(A) and C.
//	n input	number of columns of matrix op(B) and C.
//	k input number of columns of op(A) and rows of op(B).
//  lda == m
//  ldb == k
//  ldc == m
//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB,
	//d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
	int lda = width_a;
	int ldb = width_b;
	int ldc = width_b;
	cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width_b,
			height_a, width_a, &alpha, b, ldb, a, lda, &beta, c, ldc);

	if (CUBLAS_STATUS_SUCCESS != ret) {
		printf("pau no blas\n");
		exit(-1);
	}

	cublasDestroy(handle);
	return ret;
}

void matrix_multiplication_abft() {
	long lin_a = 20;//05;//96;
	long col_a = 12;//55;//48;
	long lin_b = col_a;//48;
	long col_b = 14;//02;//92;
	long vec_siz_a = ((lin_a) * (col_a));
	long vec_siz_b = ((lin_b) * (col_b));
	long vec_siz_c = ((lin_a) * (col_b));
	const long siz_a = vec_siz_a * sizeof(float);
	const long siz_b = vec_siz_b * sizeof(float);
	const long siz_c = vec_siz_c * sizeof(float);
	//host memories
	float* host_array_a = (float*) calloc(vec_siz_a, sizeof(float));
	float* host_array_b = (float*) calloc(vec_siz_b, sizeof(float));
	float* host_array_c = (float*) calloc(vec_siz_c, sizeof(float));
//	float* host_array_c_temp = (float*) calloc(vec_siz_c, sizeof(float));
	fill_mat_row_major(host_array_a, lin_a, col_a);
	fill_mat_row_major(host_array_b, lin_b, col_b);

	//cuda memories
	float *device_array_a, *device_array_b, *device_array_c;
	cudaMalloc(&device_array_a, siz_a);
	cudaMalloc(&device_array_b, siz_b);
	cudaMalloc(&device_array_c, siz_c);
	//copy to devicex_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
	cudaMemcpy(device_array_a, host_array_a, siz_a, cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_b, host_array_b, siz_b, cudaMemcpyHostToDevice);

	double time_from_host = mysecond();
	calc_checksums_from_host(device_array_a, device_array_b, lin_a, col_a, lin_b, col_b);
	printf("Calc checksums time calling from host %lf\n",
			mysecond() - time_from_host);

	cudaMemcpy(host_array_a, device_array_a, siz_a, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array_b, device_array_b, siz_b, cudaMemcpyDeviceToHost);
	print_mat_row_major(host_array_a, lin_a, col_a, "matrix A");

	print_mat_row_major(host_array_b, lin_b, col_b, "matrix B");

	dgemm_host(col_a, lin_a, col_b, lin_b, device_array_a, device_array_b,
			device_array_c);


	ErrorReturn temp = check_checksums_from_host(device_array_c, lin_a, col_b);
	cudaMemcpy(host_array_c, device_array_c, siz_c, cudaMemcpyDeviceToHost);
		time_from_host = mysecond();
	print_mat_row_major(host_array_c, lin_a, col_b, "GPU result mat");


	printf("Final check time calling from host %lf\n",
			mysecond() - time_from_host);


	printf("Detected row errors: %d\nDetected collum errors %d\n",
			temp.row_detected_errors, temp.col_detected_errors);
	printf("\n");

	gpuErrchk(cudaDeviceSynchronize());
	free(host_array_a);
	free(host_array_b);
	free(host_array_c);

	cudaFree(device_array_a);
	cudaFree(device_array_b);
	cudaFree(device_array_c);
	free_error_return(&temp);
}

int main(void) {
	matrix_multiplication_abft();
	return 0;
}
