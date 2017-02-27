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

void fill_mat_row_major(float_t *t, float_t val, long m, long n) {
	long i, j;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			t[i * n + j] = val; //((rand() % 15) / 3.14578);
}

void compare(float *t, float *s, long siz) {
	long i;
	for (i = 0; i < siz; i++) {
		if (fabs(t[i]) - fabs(s[i]) > 0.0000001)
			printf("t[%ld] is diff from s[%ld] on diff %lf", i, i,
					fabs(t[i]) - fabs(s[i]));
	}
}

void inline print_array(float_t *arr, long_t n) {
	int i;
	for (i = 0; i < n; i++)
		printf("%lf ", arr[i]);
	printf("\n");
}
void inline fill_array(float_t *arr, float_t val, long_t n) {
	int i;
	for (i = 0; i < n; i++)
		arr[i] = val;
}

void matrix_multiplication_abft() {
	long lin_a = 200;//5;//96;
	long col_a = 125;//5;//48;
	long lin_b = col_a; //48;
	long col_b = 140;//2;//92;
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
	fill_mat_row_major(host_array_a, 1.0, lin_a, col_a);
	fill_mat_row_major(host_array_b, 2.0, lin_b, col_b);

	//cuda memories
	float *device_array_a, *device_array_b, *device_array_c;
	cudaMalloc(&device_array_a, siz_a);
	cudaMalloc(&device_array_b, siz_b);
	cudaMalloc(&device_array_c, siz_c);
	//copy to devicex_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
	cudaMemcpy(device_array_a, host_array_a, siz_a, cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_b, host_array_b, siz_b, cudaMemcpyHostToDevice);

//	calc_checksums_from_host(device_array_a, device_array_b, lin_a, col_a, lin_b, col_b);

	float_t *check_col, *check_row;
	float_t *h_check_col, *h_check_row;

	cudaMalloc(&check_col, col_b * sizeof(float_t));
	cudaMalloc(&check_row, lin_a * sizeof(float_t));
	h_check_col = (float*) calloc(col_b, sizeof(float));
	h_check_row = (float*) calloc(col_a, sizeof(float));

	float_t *dev_mat;

	long_t max = col_a;
	if (lin_a > col_a)
		max = lin_a;

	if (col_b > lin_a)
		max = col_b;
	printf("max %ld\n", max);

	cudaMalloc(&dev_mat, max * sizeof(float_t));
	float_t *h_dev_mat = (float*) calloc(max, sizeof(float));

	fill_array(h_dev_mat, 1.0, max);

	cudaMemcpy(dev_mat, h_dev_mat, sizeof(float_t) * max,
			cudaMemcpyHostToDevice);

	int i = 0;
	double time_from_host = mysecond();
	for (i = 0; i < 10; i++) {
		time_from_host = mysecond();
		calc_checksums(device_array_a, device_array_b, dev_mat, check_row,
				check_col, lin_a, col_a, col_b);
		printf("Calc checksums time calling from host %lf\n",
				mysecond() - time_from_host);
	}
	cudaMemcpy(host_array_a, device_array_a, siz_a, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array_b, device_array_b, siz_b, cudaMemcpyDeviceToHost);

	print_mat_row_major(host_array_a, lin_a, col_a, "matrix A");

	print_mat_row_major(host_array_b, lin_b, col_b, "matrix B");

	cudaMemcpy(h_check_col, check_col, col_b * sizeof(float_t),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(h_check_row, check_row, lin_a * sizeof(float_t),
			cudaMemcpyDeviceToHost);

	printf("Vetor saida colunas\n");
	print_array(h_check_col, col_b);

	printf("Vetor saida linhas\n");
	print_array(h_check_row, lin_a);

	//move the data to a new matrix
//	float_t *dev_mat_a_aux, *dev_mat_b_aux, *dev_mat_c_aux;

//	long vec_siz_a_aux = ((lin_a + 1) * (col_a));
//	long vec_siz_b_aux = ((lin_b) * (col_b + 1));
//	long vec_siz_c_aux = ((lin_a + 1) * (col_b + 1));
//	cudaMalloc(&dev_ma_a_aux, vec_siz_a_aux * sizeof(float_t));
//	cudaMalloc(&dev_ma_b_aux, vec_siz_b_aux * sizeof(float_t));
//	cudaMalloc(&dev_ma_c_aux, vec_siz_c_aux * sizeof(float_t));

	dgemm_host(col_a, lin_a, col_b, lin_b, device_array_a, device_array_b,
			device_array_c);

//	ErrorReturn temp = check_checksums_from_host(device_array_c, lin_a, col_b);
	cudaMemcpy(host_array_c, device_array_c, siz_c, cudaMemcpyDeviceToHost);
	time_from_host = mysecond();

	print_mat_row_major(host_array_c, lin_a, col_b, "GPU result mat");

	printf("Final check time calling from host %lf\n",
			mysecond() - time_from_host);

//	printf("Detected row errors: %d\nDetected collum errors %d\n",
//			temp.row_detected_errors, temp.col_detected_errors);
	printf("\n");

	gpuErrchk(cudaDeviceSynchronize());
	free(host_array_a);
	free(host_array_b);
	free(host_array_c);

	cudaFree(device_array_a);
	cudaFree(device_array_b);
	cudaFree(device_array_c);

//	free(h_check_col);
//	free(h_check_row);
//	cudaFree(dev_mat);
//	cudaFree(check_col);
//	cudaFree(check_row);
//	free_error_return(&temp);
}

int main(void) {
	matrix_multiplication_abft();
	return 0;
}
