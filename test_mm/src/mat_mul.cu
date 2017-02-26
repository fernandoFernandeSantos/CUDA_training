#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
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
//	calc_checksums_from_host(device_array_a, device_array_b, lin_a, col_a, lin_b, col_b);
	printf("Calc checksums time calling from host %lf\n",
			mysecond() - time_from_host);

	cudaMemcpy(host_array_a, device_array_a, siz_a, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array_b, device_array_b, siz_b, cudaMemcpyDeviceToHost);
	print_mat_row_major(host_array_a, lin_a, col_a, "matrix A");

	print_mat_row_major(host_array_b, lin_b, col_b, "matrix B");


	float_t *check_col, *check_row;
	float_t *h_check_col, *h_check_row;


	cudaMalloc(&check_col, col_b * sizeof(float_t));
	cudaMalloc(&check_row, lin_a * sizeof(float_t));
	h_check_col = (float*) calloc(col_b, sizeof(float));
	h_check_row = (float*) calloc(col_a, sizeof(float));

	float_t *dev_mat;
//	calc_checksums(device_array_a, device_array_b, dev_mat, check_row, check_col, lin_a, col_a, col_b);
	long_t max = col_a;
	if (lin_a > col_a)
		max = lin_a;

	if (col_b > lin_a)
		max = col_b;
	printf("max %ld\n", max);
	cudaMalloc(&dev_mat, max * sizeof(float_t));
	cudaMemset(dev_mat, 1, max * sizeof(float_t));

	cudaMemcpy(h_check_col, check_col, col_b * sizeof(float_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_check_row, dev_mat, col_a * sizeof(float_t), cudaMemcpyDeviceToHost);

	int i = 0;
	printf("Vetor saida colunas\n");
	for(i = 0; i < col_b; i++){
		printf("%lf ", h_check_col[i]);
	}

	printf("\n");
	printf("Vetor saida linhas\n");
	for(i =0; i < col_a; i++){
		printf("%lf ", h_check_row[i]);
	}

	printf("\n");

	cudaFree(dev_mat);
	cudaFree(check_col);
	cudaFree(check_row);


	//
//	dgemm_host(col_a, lin_a, col_b, lin_b, device_array_a, device_array_b,
//			device_array_c);


	ErrorReturn temp = check_checksums_from_host(device_array_c, lin_a, col_b);
	cudaMemcpy(host_array_c, device_array_c, siz_c, cudaMemcpyDeviceToHost);
		time_from_host = mysecond();
//	print_mat_row_major(host_array_c, lin_a, col_b, "GPU result mat");


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
