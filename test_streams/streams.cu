#include <pthread.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "cuda.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"

typedef float Real;

__global__ void kernel(float *x, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
		float sum = 0;
		for (int j = 1; j < n; j++) {
			sum += sqrt(pow(3.14159, i)) / j;
		}
		x[i] = sum;
	}
}

void *launch_kernel(void *dummy) {
	float *data;
	int N = 1 << 20;

	cudaMalloc(&data, N * sizeof(float));
	kernel<<<1, 64>>>(data, N);

	cudaStreamSynchronize(0);

	return NULL;
}

///////////////////////////////////////////
//Matrix multiplication pthreads

typedef struct {
	float *a_device;
	float *b_device;
	float *c_device;
	int a_col_size;
	int a_lin_size;

	int b_col_size;
	int b_lin_size;

} thread_parameters;

void *launch_sgemm(void *data) {
	thread_parameters *parameter = (thread_parameters*) data;

	cudaStream_t stream;
	cublasHandle_t handle;
	cudaStreamCreate(&stream);
	cublasCreate(&handle);
	cublasSetStream(handle, stream);
	int lda = parameter->a_col_size;
	int ldb = parameter->b_col_size;
	int ldc = parameter->b_col_size;
	Real alpha = 1.0f;
	Real beta = 0.0f;
	cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			parameter->b_col_size, parameter->a_lin_size, parameter->a_col_size,
			&alpha, parameter->b_device, ldb, parameter->a_device, lda, &beta,
			parameter->c_device, ldc);

	if (CUBLAS_STATUS_SUCCESS != ret) {
		printf("pau no blas %d\n", ret);
		exit(-1);
	}

	cublasDestroy(handle);
	cudaStreamSynchronize(0);

	return NULL;
}

thread_parameters fill_data(int a_lin, int a_col, int b_col) {
	thread_parameters ret;
	ret.a_col_size = a_col;
	ret.a_lin_size = a_lin;
	ret.b_col_size = b_col;
	ret.b_lin_size = ret.a_col_size;

	Real *host_a = (Real*) malloc(a_col * a_lin * sizeof(Real));
	Real *host_b = (Real*) malloc(b_col * a_col * sizeof(Real));

	for (int i = 0; i < a_col * a_lin; i++)
		host_a[i] = random() % 200;

	for (int i = 0; i < b_col * a_col; i++)
		host_b[i] = random() % 100;

	cudaMalloc(&ret.a_device, sizeof(Real) * (ret.a_col_size * ret.a_lin_size));
	cudaMalloc(&ret.b_device, sizeof(Real) * (ret.b_col_size * ret.b_lin_size));
	cudaMalloc(&ret.c_device,
			sizeof(Real) * (ret.a_lin_size) * (ret.b_col_size));

	cudaMemcpy(ret.a_device, host_a, a_col * a_lin, cudaMemcpyHostToDevice);
	cudaMemcpy(ret.b_device, host_b, b_col * a_col, cudaMemcpyHostToDevice);
	free(host_a);
	free(host_b);
	return ret;
}

void free_data(thread_parameters data) {
	cudaFree(data.a_device);
	cudaFree(data.b_device);
	cudaFree(data.c_device);
	data.a_col_size = data.a_lin_size = data.b_col_size = data.b_lin_size = 0;
}

int main() {
	const int num_threads = 8;

	pthread_t threads[num_threads];
	thread_parameters data[num_threads];
	for (int i = 0; i < num_threads; i++)
		data[i] = fill_data(2048, 2048, 2048);

	for (int i = 0; i < num_threads; i++) {

		if (pthread_create(&threads[i], NULL, launch_sgemm, &data[i])) {
			fprintf(stderr, "Error creating threadn");
			return 1;
		}
	}

	for (int i = 0; i < num_threads; i++) {
		if (pthread_join(threads[i], NULL)) {
			fprintf(stderr, "Error joining threadn");
			return 2;
		}
		free_data(data[i]);
	}

	cudaDeviceReset();

	return 0;
}
