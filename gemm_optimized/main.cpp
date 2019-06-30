/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include "device_vector.h"
#include "sgemm_nn_64_16_16_16_4.h"
#include <cassert>
#include <vector>
#include <iostream>


int main(int argc, char **argv) {

	int m;
	int n;
	int k;
	m = n = k = 8192;
	int lda = m;
	int ldb = n;
	int ldc = k;
	float alpha = 0.1;
	float beta = 0.3;

	std::vector<float> host_a(m * n, alpha);
	std::vector<float> host_b(n * k, beta);
	std::vector<float> host_c(m * k, 0.0);

	rad::DeviceVector<float> device_c(host_c), device_a(host_a), device_b(host_b);

	cudaStream_t st;
	cudaStreamCreate(&st);
	assert(m > 512 && n > 512 && m % 64 == 0 && n % 16 == 0 && k % 16 == 0);
	sgemm_N_N_64_16_16_16_4_special(st, device_c.data(), device_a.data(), device_b.data(), m, n, k,
					lda, ldb, ldc, alpha, beta);

	host_c = device_c.to_vector();

	for(int i = 0; i < 10; i++){
		for(int j = 0; j < 10; j++){
			std::cout << host_c[i * m + j] << " ";
		}
		std::cout << std::endl;
	}

	cudaStreamDestroy(st);
	return 0;
}
