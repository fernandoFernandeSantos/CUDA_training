/*
 * cuda_lock.h
 *
 *  Created on: 11/06/2019
 *      Author: fernando
 */

#ifndef CUDA_LOCK_H_
#define CUDA_LOCK_H_

struct Lock {
	int mutex;

	Lock() :
			mutex(0) {
//		checkCudaErrors(cudaMalloc(&mutex, sizeof(int)));
//		cudaMemcpy(mutex, &state, sizeof(int))

	}

	__device__ void lock() {
		while (atomicCAS(&this->mutex, 0, 1) != 0)
			;
	}

	__device__ void unlock() {
		atomicExch(this->mutex, 0);
	}
};

#endif /* CUDA_LOCK_H_ */
