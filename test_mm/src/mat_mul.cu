#include <stdio.h>
#include <stdlib.h>

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

#define BLOCK_SIZE 1024

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

__global__ void check_col(float *mat, long rows, long cols) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	float acc = 0;
	//must be less one
	for (k = 0; k < cols - 1; k++) {
		acc += mat[i * cols + k];
	}
	long b_index = i * cols + cols - 1;
	//printf("b_index %ld acc %lf \n", b_index, acc);
	float diff = fabs(fabs(mat[b_index]) - fabs(acc));
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&col_detected_errors, 1);
//		printf("passou no col mat[%ld] = %ld diff %ld calc %ld i %ld\n",
//				b_index, (long) mat[b_index], (long) diff, (long) acc, i);
	}
	//__syncthreads();
}

__global__ void check_row(float *mat, long rows, long cols) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	float acc = 0;
	//must be less one
	for (k = 0; k < rows - 1; k++) {
		acc += mat[k * cols + j];
	}
	//printf("a_index %ld acc %lf \n", rows_a * cols_a + j, acc);
	long a_index = (rows - 1) * cols + j;
	float diff = fabs(fabs(mat[a_index]) - fabs(acc));
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&row_detected_errors, 1);
//		printf("passou no row mat[%ld] = %ld diff %ld calc %ld i value %ld\n",
//				a_index, (long) mat[a_index - 1], (long) diff, (long) acc, j);
	}
	//__syncthreads();
}

//DYNAMIC PARALLELISM ONLY TO CALL NEW KERNELS, ARE FUCK KIDDING???
//man, I am so lazy
__global__ void check_checksums(float *c, long rows_c, long cols_c) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("i value %ld\n", i);
	//rows
	if (i == 0) {
		long blocks = ceil(float(cols_c) / float(BLOCK_SIZE));
		long threads = ceil(float(cols_c) / float(blocks));
		check_row<<<blocks, threads>>>(c, rows_c, cols_c);
//		printf("cols %d blocks %ld threads %ld\n", cols_c, blocks, threads);
	}
	//cols
	if (i == 1) {
		long blocks = ceil(float(rows_c) / float(BLOCK_SIZE));
		long threads = ceil(float(rows_c) / float(blocks));
		check_col<<<blocks, threads>>>(c, rows_c, cols_c);
		printf("blocks %ld threads %ld\n", blocks, threads);
	}
	//printf("passou aqui foi\n");

	__syncthreads();
	//printf("values %d %d\n ", row_detected_errors, col_detected_errors);
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

	long k;
	float acc = 0;
	for (k = 0; k < rows_a - 1; k++) {
		acc += a[k * cols_a + j];
	}

	long a_index = (rows_a - 1) * cols_a + j;
	//printf("a_index %ld acc %lf \n", a_index, acc);
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

	long k;
	float acc = 0;
	for (k = 0; k < cols_b - 1; k++) {
		acc += b[i * cols_b + k];
	}
	long b_index = i * cols_b + cols_b - 1;
	//if (i == 0)	b[1] = 9999; //pseudo fault injection

	//printf("b_index %ld acc %lf \n", b_index, acc);

	b[b_index] = acc;
}

__global__ void calc_checksums(float *a, float *b, long rows_a, long cols_a,
		long rows_b, long cols_b) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("i value %ld\n", i);
	//rows
	if (i == 0) {
		//1d grid for abft operations
		long blocks = ceil(float(cols_a) / float(BLOCK_SIZE));
		long threads = ceil(float(cols_a) / float(blocks));
		first_abraham_op<<<blocks, threads>>>(a, rows_a, cols_a);
	}

	if (i == 1) {
		//second
		long blocks = ceil(float(rows_b) / float(BLOCK_SIZE));
		long threads = ceil(float(rows_b) / float(blocks));
		second_abraham_op<<<blocks, threads>>>(b, rows_b, cols_b);
	}
	__syncthreads();
}

void abraham_sum(float *a, float *b, long rows_a, long cols_a, long rows_b,
		long cols_b) {
	calc_checksums<<<1, 2>>>(a, b, rows_a, cols_a, rows_b, cols_b);
	gpuErrchk(cudaPeekAtLastError());
}

void abraham_check(float *c, long rows, long cols) {
	check_checksums<<<1, 2>>>(c, rows, cols);
	gpuErrchk(cudaPeekAtLastError());
}

void print_mat_row_major(float *mat, long m, long n, const char *mat_name) {
	printf("ROW-MAJOR ORDER: printing %s lin %ld col %ld\n", mat_name, m, n);
	long i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++)
			printf("%ld ", (PRINT_TYPE) mat[i * n + j]);
		printf("\n");
	}
//	printf("on vector 1d\n");
//	for (i = 0; i < m * n; i++) {
//		printf("%ld ", (PRINT_TYPE) mat[i]);
//	}
	printf("\n");
}

void print_mat_collum_major(float *mat, long m, long n, const char *mat_name) {
	printf("COLLUM-MAJOR ORDER: printing %s lin %ld col %ld\n", mat_name, m, n);
	long i, j;
	for (i = 0; i < m; i++) {

		for (j = 0; j < n; j++) {
			printf("%ld ", (PRINT_TYPE) mat[j * m + i]);
		}
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
			t[i * n + j] = 1; //float(i);
}

void fill_mat_collum_major(float *t, long m, long n) {
	long i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			t[j * m + i] = float(i);
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

	cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, height_b, width_a, width_b, &alpha, b, width_b, a, width_a, &beta, c,
			width_b);

	if (CUBLAS_STATUS_SUCCESS != ret) {
		printf("pau no blas\n");
		exit(-1);
	}

	cublasDestroy(handle);
	return ret;
}

void matrix_multiplication_abft() {
	long lin_a = 2048;
	long col_a = 2048;
	long lin_b = 2048;
	long col_b = 2048;
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

	abraham_sum(device_array_a, device_array_b, lin_a, col_a, lin_b, col_b);

	cudaMemcpy(host_array_a, device_array_a, siz_a, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array_b, device_array_b, siz_b, cudaMemcpyDeviceToHost);
//	print_mat_row_major(host_array_a, lin_a, col_a, "matrix A");
//	printf("\n");
//	print_mat_row_major(host_array_b, lin_b, col_b, "matrix B");

	dgemm_host(col_a, lin_a, col_b, lin_b, device_array_a, device_array_b,
			device_array_c);

	cudaMemcpy(host_array_c, device_array_c, siz_c, cudaMemcpyDeviceToHost);
//	print_mat_row_major(host_array_c, lin_a, col_b, "GPU result mat");
	int row_detected_errors_host = 0, col_detected_errors_host = 0;

	abraham_check(device_array_c, (lin_a), (col_b));

	cudaMemcpyFromSymbol(&row_detected_errors_host, row_detected_errors,
			sizeof(int));
	cudaMemcpyFromSymbol(&col_detected_errors_host, col_detected_errors,
			sizeof(int));
	printf("Detected row errors: %d\nDetected collum errors %d\n",
			row_detected_errors_host, col_detected_errors_host);

	gpuErrchk(cudaDeviceSynchronize());
	free(host_array_a);
	free(host_array_b);
	free(host_array_c);

	cudaFree(device_array_a);
	cudaFree(device_array_b);
	cudaFree(device_array_c);
}

int main(void) {
	matrix_multiplication_abft();
	return 0;
}
