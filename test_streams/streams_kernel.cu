/*
 * streams_kernel.cu
 *
 *  Created on: 14/02/2018
 *      Author: fernando
 */
#include <pthread.h>
#include <stdio.h>

__global__ void sqrt_streams(float *x, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
		float sum = 0;
		for (int j = 1; j < 100; j++){
			sum += sqrt(pow(3.14159, i))/j;
		}
		x[i] = sum;
	}
}


void vector_streams() {
	const int num_streams = 8;
	const int N = 1 << 20;
	cudaStream_t streams[num_streams];
	float *data[num_streams];

	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);

		cudaMalloc(&data[i], N * sizeof(float));

		// launch one worker kernel per stream
		sqrt_streams<<<1, 64, 0, streams[i]>>>(data[i], N);

		sqrt_streams<<<1, 1>>>(0, 0);

	}

	for (int i = 0; i < num_streams; i++)
		cudaFree(data[i]);
	cudaDeviceReset();
}

int main(int argc, char **argv) {
	vector_streams();
	return 0;
}
