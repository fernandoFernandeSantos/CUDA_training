#include <pthread.h>
#include <stdio.h>

const int N = 1 << 20;

__global__ void kernel(float *x, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
		float sum = 0;
		for (int j = 1; j < n; j++){
			sum += sqrt(pow(3.14159, i))/j;
		}
		x[i] = sum;
	}
}

void *launch_kernel(void *dummy) {
	float *data;
	cudaMalloc(&data, N * sizeof(float));

	kernel<<<1, 64>>>(data, N);

	cudaStreamSynchronize(0);

	return NULL;
}


///////////////////////////////////////////
//Matrix multiplication pthreads

typedef struct{
	float *a_device;
	float *b_device;
	float *c_device;
	int a_col_size;
	int a_lin_size;

	int b_col_size;
	int b_lin_size;

}thread_parameters;


void *launch_sgemm(void *data){
	thread_parameters *parameter = (thread_parameters*)data;

	printf("Thread Id: %d a_col %d a_lin %d b_col %d b_lin %d\n", pthread_self(), parameter->a_col_size, parameter->a_lin_size,
			parameter->b_col_size, parameter->b_lin_size);

	return NULL;
}

int main() {
	const int num_threads = 8;

	pthread_t threads[num_threads];

	for (int i = 0; i < num_threads; i++) {
		if (pthread_create(&threads[i], NULL, launch_sgemm, 0)) {
			fprintf(stderr, "Error creating threadn");
			return 1;
		}
	}

	for (int i = 0; i < num_threads; i++) {
		if (pthread_join(threads[i], NULL)) {
			fprintf(stderr, "Error joining threadn");
			return 2;
		}
	}

	cudaDeviceReset();

	return 0;
}
