/*
 * persistent_lib.h
 *
 *  Created on: 14/06/2019
 *      Author: fernando
 */

#ifndef PERSISTENT_LIB_H_
#define PERSISTENT_LIB_H_

#define MAXTHREADNUMBER 2048

typedef unsigned long long uint64;
typedef unsigned int uint32;
typedef unsigned char byte;

__device__ bool running;
__device__ byte thread_flags[MAXTHREADNUMBER];

__device__ static void sleep_cuda(uint64 clock_count) {
	uint64 start = clock64();
	uint64 clock_offset = 0;
	while (clock_offset < clock_count) {
		clock_offset = clock64() - start;
	}
}

struct HostPersistentControler {
	cudaStream_t st;
	uint64 thread_number;
	std::vector<byte> host_thread_flags;
	const std::vector<byte> zero_vector;

	HostPersistentControler(uint64 thread_number) :
			thread_number(thread_number), zero_vector(
					std::vector < byte > (MAXTHREADNUMBER, 0)) {
		checkCudaErrors(
				cudaStreamCreateWithFlags(&this->st, cudaStreamNonBlocking));

		this->set_running(true);
		this->host_thread_flags = std::vector < byte > (MAXTHREADNUMBER, 0);
	}

	virtual ~HostPersistentControler() {
		checkCudaErrors(cudaStreamDestroy(this->st));
	}

	void end_kernel() {
		this->set_running(false);
	}

	void start_processing() {
		checkCudaErrors(
				cudaMemcpyToSymbolAsync(thread_flags, this->zero_vector.data(),
				MAXTHREADNUMBER, 0, cudaMemcpyHostToDevice, st));
		checkCudaErrors(cudaStreamSynchronize(st));
	}

	void wait_gpu() {
		while (true) {
			checkCudaErrors(
					cudaMemcpyFromSymbolAsync(this->host_thread_flags.data(),
							thread_flags,
							MAXTHREADNUMBER, 0, cudaMemcpyDeviceToHost, st));
			checkCudaErrors(cudaStreamSynchronize(st));
			int counter = 0;
			for (auto bt : host_thread_flags) {
				if (bt == 1)
					counter++;
			}
			std::cout << "FINISHED " << counter << " " << this->thread_number
					<< std::endl;

			if (this->thread_number <= counter) {
				return;
			}
			sleep(1);
		}
	}

private:
	void set_running(bool value) {
		std::cout << "Setting running to " << value << std::endl;
		checkCudaErrors(
				cudaMemcpyToSymbolAsync(running, &value, sizeof(bool), 0,
						cudaMemcpyHostToDevice, st));
		checkCudaErrors(cudaStreamSynchronize(st));

		std::cout << "Running set to " << value << std::endl;
	}
};

struct PersistentKernel {
	bool& running_;
	uint32 thread_id;

	__device__ PersistentKernel() :
			running_(running) {
		this->thread_id = get_global_idx();
	}

	__device__ void wait_for_work() {
		while (thread_flags[this->thread_id] == 1) {
		}
		//printf("ATOMIC %d %d\n", thread_flags[this->thread_id], this->thread_id);

	}

	__device__ uint32 get_global_idx() {
		uint32 blockId = blockIdx.x + blockIdx.y * gridDim.x
				+ gridDim.x * gridDim.y * blockIdx.z;

		uint32 threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
				+ (threadIdx.z * (blockDim.x * blockDim.y))
				+ (threadIdx.y * blockDim.x) + threadIdx.x;
//		printf("THREAD ID %d\n", threadId);

		return threadId;
	}

	__device__ void iteration_finished() {
		thread_flags[this->thread_id] = 1;
	}

	__device__ bool keep_working() {
		return this->running_;
	}

};

#endif /* PERSISTENT_LIB_H_ */
