#include "cuda.h"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>

#define BLOCK_SIZE 1024
#define MAX_BLOCKS 65535
#define MAX_GB 2

typedef double memory;
typedef unsigned char byte_memory;
typedef unsigned long long int counter_type;

__global__ void mem_compare(memory *mem, const memory constant,
		size_t array_size, counter_type *counter) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	//mem[55] = 0xbb;
//	if (i == array_size - 1)
//		printf("Passou\n");
	if (i < array_size) {
		//register memory temp = *(mem + i) ^ constant;
		if (mem[i] != constant) {
			//printf("%d\n", i);
			atomicAdd(counter, 1);
			//                        printf("%d\n", counter);

		}
	}
}

__global__ void memory_set(memory *mem, const memory constant,
		size_t array_size) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < array_size) {
		mem[i] = constant;
	}
}

void check_framework_errors(cudaError_t error) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	printf("CUDA Framework error: %s. Bailing.", cudaGetErrorString(error));

	printf("%s\n", errorDescription);
	exit(EXIT_FAILURE);
}

void host_to_gpu_compare(int giga, const long conversion) {
	// if it is more than 2GB
	long host_slices = giga;
	long size_for_host, size;
	if (giga > MAX_GB) {
		host_slices = ceil(float(giga) / float(MAX_GB));
		size_for_host = MAX_GB * conversion;
		// must allocate all the sizes equal
		size = host_slices * size_for_host;
	} else {
		host_slices = 1;
		size_for_host = size = giga * conversion;
	}
	//------------------------------------------------------
	printf("Testing for %lu bytes, and %dGB\n", size, giga);
	byte_memory *d_a, *h_a, *gold_a;
	check_framework_errors(cudaMalloc((void**) (&d_a), size));
	h_a = (byte_memory*) (malloc(size_for_host));
	gold_a = (byte_memory*) (malloc(size_for_host));
	printf("Memory allocation finished\n");
	if ((h_a == NULL) || (gold_a == NULL)) {
		printf("error host malloc\n");
		exit(EXIT_FAILURE);
	}
	// ===> FIRST PHASE: CHECK SETTING BITS TO 10101010
	check_framework_errors(cudaMemset((void**) (d_a), 0xAA, size));
	memset(gold_a, 0xAA, size_for_host);
	int err_count_AA = 0, error_count_55 = 0;
	for (int i = 0; i < host_slices; i++) {
		printf("Host iteration %d size of processing %ld host slices %ld\n", i,
				(i * size_for_host), host_slices);
		check_framework_errors(
				cudaMemcpy(h_a, d_a + (i * size_for_host), size_for_host,
						cudaMemcpyDeviceToHost));
		for (long i = 0; i < size_for_host; i++) {
			if (h_a[i] != gold_a[i]) {
				err_count_AA++;
			}
		}
		// ===> END FIRST PHASE
	}
	// ===> SECOND PHASE: CHECK SETTING BITS TO 01010101
	check_framework_errors(cudaMemset(d_a, 0x55, size));
	memset(gold_a, 0x55, size_for_host);
	for (int i = 0; i < host_slices; i++) {
		printf("Host iteration %d size of processing %ld host slices %ld\n", i,
				(i * size_for_host), host_slices);
		check_framework_errors(
				cudaMemcpy(h_a, d_a + (i * size_for_host), size_for_host,
						cudaMemcpyDeviceToHost));
		for (long i = 0; i < size_for_host; i++) {
			if (h_a[i] != gold_a[i]) {
				error_count_55++;
			}
		}
	}
	printf(
			"Total errors with 0xAA setting: %d\nTotal errors with 0x55 setting: %d\n",
			err_count_AA, error_count_55);
	// ===> END SECOND PHASE
	free(gold_a);
	free(h_a);
	cudaFree(d_a);
}

void gpu_mem_check(float giga, int iterations, const size_t conversion) {
	const memory constant_types[4] = { 3.22229884838849e-5, 5.76546454698898e-5,
			7.2345678899997564e-5, 8.122344565443123e-5 };
	//{ 0x55, 0xaa, 0xFF, 0x00 };
	const size_t size = (size_t) ((giga * float(conversion)));
	memory* device_memory;
	counter_type* counter;
	check_framework_errors(cudaMalloc((void**) ((&device_memory)), size));
	check_framework_errors(
			cudaMalloc((void**) ((&counter)), sizeof(counter_type)));
	dim3 threads(BLOCK_SIZE), blocks = ceil(
			float(size / sizeof(memory)) / float(BLOCK_SIZE));
	std::cout << "Size of vector " << size / sizeof(memory) << " block.x "
			<< blocks.x << " threads.x " << threads.x << std::endl;
	//memcomparing on gpu
	for (int i = 0; i < iterations; i++) {
		for (int j = 0; j < 4; j++) {
			//set counter to 0
			check_framework_errors(
					cudaMemset(counter, 0, sizeof(counter_type)));
						check_framework_errors(
								cudaMemset(device_memory, constant_types[j], size / sizeof(memory)));
//			memory_set<<<blocks, threads>>>(device_memory, constant_types[j],
//					size / sizeof(memory));
			check_framework_errors(cudaDeviceSynchronize());
			//compare memory
			mem_compare<<<blocks, threads>>>(device_memory, constant_types[j],
					size / sizeof(memory), counter);
			check_framework_errors(cudaDeviceSynchronize());
			//copy counter back
			counter_type host_counter = 0;
			check_framework_errors(
					cudaMemcpy(&host_counter, counter, sizeof(counter_type),
							cudaMemcpyDeviceToHost));
			std::cout << "Size " << size << " memory errors for "
					<< constant_types[j] << " config are " << host_counter
					<< std::endl;
		}
	}
	check_framework_errors(cudaFree(device_memory));
	check_framework_errors(cudaFree(counter));
}

int main(int argc, char** argv) {
	std::string option(argv[1]);

	float giga = atof(argv[2]);
	int iterations = 10;
	if (argc > 3) {
		iterations = atoi(argv[3]);
	}

	const size_t conversion = 1024 * 1024 * 1024;


	if(option == "host"){
		host_to_gpu_compare(giga, conversion);
	}else if (option == "gpu"){
		gpu_mem_check(giga, iterations, conversion);
	}
	return 0;

}
