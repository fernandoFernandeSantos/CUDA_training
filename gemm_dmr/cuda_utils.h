/*
 * cuda_utils.h
 *
 *  Created on: 27/03/2019
 *      Author: fernando
 */

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <sys/time.h>
#include <stdio.h>


#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)

void __checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}

	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));

	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}


#define checkBlasFrameworkErrors(error) __checkBlasFrameworkErrors(error, __LINE__, __FILE__)

void __checkBlasFrameworkErrors(cublasStatus_t status, int line, const char* file) {
	if (status == CUBLAS_STATUS_SUCCESS) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA cuBLAS Framework error: %d. Bailing.",
			status);

	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}


cudaDeviceProp GetDevice() {
//================== Retrieve and set the default CUDA device
	cudaDeviceProp prop;
	int count = 0;

	checkFrameworkErrors(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		checkFrameworkErrors(cudaGetDeviceProperties(&prop, i));
	}
	int *ndevice;
	int dev = 0;
	ndevice = &dev;
	checkFrameworkErrors(cudaGetDevice(ndevice));

	checkFrameworkErrors(cudaSetDevice(0));
	checkFrameworkErrors(cudaGetDeviceProperties(&prop, 0));

	return prop;
}

double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}



#endif /* CUDA_UTILS_H_ */
