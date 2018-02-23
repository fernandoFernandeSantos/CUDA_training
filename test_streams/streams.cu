#include <pthread.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "cuda.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"

#define BLOCK_SIZE 32

typedef float Real;

inline void __cudaSafeCall(cudaError err, const char *file, const int line);
inline void __cudaCheckError(const char *file, const int line,
		cudaStream_t stream);

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError(stream)    __cudaCheckError( __FILE__, __LINE__ , stream)

void cuda_gridsize(dim3 *threads, dim3 *blocks, size_t x, size_t y = 1,
		size_t z = 1);

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {

	if (cudaSuccess != err) {
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

inline void __cudaCheckError(const char *file, const int line,
		cudaStream_t stream) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaStreamSynchronize(stream);

	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

__global__ void kernel(float *x, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
		double sum = 0;
		for (int j = 0; j < 1000; j++) {
			sum += sqrt(pow(3.14159, i)) / float(j);
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

//1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
void cuda_gridsize(dim3 *threads, dim3 *blocks, size_t x, size_t y, size_t z) {
	int true_block_size = BLOCK_SIZE;
	if (y == 1 && z == 1)
		true_block_size = BLOCK_SIZE * BLOCK_SIZE;

	long blocks_x = ceil(float(x) / float(true_block_size));
	long threads_x = ceil(float(x) / float(blocks_x));
	long blocks_y = ceil(float(y) / float(true_block_size));
	long threads_y = ceil(float(y) / float(blocks_y));
	long blocks_z = ceil(float(z) / float(true_block_size));
	long threads_z = ceil(float(z) / float(blocks_z));

	*blocks = dim3(blocks_x, blocks_y, blocks_z);
	*threads = dim3(threads_x, threads_y, threads_z);

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
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cublasCreate(&handle);
	cublasSetStream(handle, stream);
	int lda = parameter->a_col_size;
	int ldb = parameter->b_col_size;
	int ldc = parameter->b_col_size;
	Real alpha = 1.0f;
	Real beta = 0.0f;

	dim3 threads;
	dim3 blocks;
	cuda_gridsize(&threads, &blocks,
			parameter->b_col_size * parameter->b_lin_size, 1, 1);

	for (int i = 0; i < 10; i++) {
		cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				parameter->b_col_size, parameter->a_lin_size,
				parameter->a_col_size, &alpha, parameter->b_device, ldb,
				parameter->a_device, lda, &beta, parameter->c_device, ldc);

		kernel<<<blocks, threads, 0, stream>>>(parameter->b_device,
				parameter->b_col_size * parameter->b_lin_size);
		CudaCheckError(stream);
		if (CUBLAS_STATUS_SUCCESS != ret) {
			printf("pau no blas %d\n", ret);
			exit(-1);
		}
	}

	cublasDestroy(handle);
	cudaStreamDestroy(stream);
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
	const int num_threads = 4;

	pthread_t threads[num_threads];
	thread_parameters data[num_threads];
	for (int i = 0; i < num_threads; i++)
		data[i] = fill_data(256, 256, 512);

	for (int i = 0; i < num_threads; i++) {

		if (pthread_create(&threads[i], NULL, launch_sgemm, &data[i])) {
			fprintf(stderr, "Error creating thread\n");
			return 1;
		}
	}

	for (int i = 0; i < num_threads; i++) {
		if (pthread_join(threads[i], NULL)) {
			fprintf(stderr, "Error joining thread\n");
			return 2;
		}
		free_data(data[i]);
	}

	cudaDeviceReset();

	return 0;
}
