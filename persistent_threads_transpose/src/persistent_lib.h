/*
 * persistent_lib.h
 *
 *  Created on: 14/06/2019
 *      Author: fernando
 */

#ifndef PERSISTENT_LIB_H_
#define PERSISTENT_LIB_H_

#define MAXTHREADNUMBER 2048

#define UINTCAST(x) ((unsigned int*)(x))

typedef unsigned long long uint64;
typedef unsigned int uint32;

typedef unsigned char byte;

volatile __device__ byte running;
//__device__      volatile byte thread_flags[MAXTHREADNUMBER];

volatile __device__ uint32 gpu_mutex;

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
//	std::vector<byte> host_thread_flags;
//	const std::vector<byte> zero_vector;

	HostPersistentControler(uint64 thread_number) :
			thread_number(thread_number) {
		checkCudaErrors(
				cudaStreamCreateWithFlags(&this->st, cudaStreamNonBlocking));

		this->set_running(1);
//		this->host_thread_flags = std::vector < byte > (MAXTHREADNUMBER, 0);
	}

	virtual ~HostPersistentControler() {
		checkCudaErrors(cudaStreamDestroy(this->st));
	}

	void end_kernel() {
		this->set_running(0);
	}

	void start_processing() {
		uint32 tmp = 0;
		checkCudaErrors(
				cudaMemcpyToSymbolAsync(UINTCAST(gpu_mutex), UINTCAST(tmp), sizeof(uint32), 0,
						cudaMemcpyHostToDevice, st));
		checkCudaErrors(cudaStreamSynchronize(st));
	}

	void wait_gpu() {
		while (true) {
			uint32 counter;
			checkCudaErrors(
					cudaMemcpyFromSymbolAsync(&counter, gpu_mutex,
							sizeof(uint32), 0, cudaMemcpyDeviceToHost, st));
			checkCudaErrors(cudaStreamSynchronize(st));

			std::cout << "FINISHED " << counter << " " << this->thread_number
					<< std::endl;

			if (this->thread_number <= counter) {
				return;
			}
			sleep(1);
		}
	}

private:
	void set_running(byte value) {
		std::cout << "Setting running to " << value << std::endl;
		checkCudaErrors(
				cudaMemcpyToSymbolAsync(running, &value, sizeof(byte), 0,
						cudaMemcpyHostToDevice, st));
		checkCudaErrors(cudaStreamSynchronize(st));

		std::cout << "Running set to " << value << std::endl;
	}
};

struct PersistentKernel {
//	uint32 thread_id;
	uint32 blocks_to_synch;
	bool local_process;
	uint32 tid_in_block;

	__device__ PersistentKernel()
//	:	thread_id(get_global_idx())
	{
		//thread ID in a block
		this->tid_in_block = this->get_block_idx();
		this->blocks_to_synch = gridDim.x * gridDim.y * gridDim.z;
		this->local_process = false;

	}

	__device__ void wait_for_work() {
		while (this->local_process) {
			if (gpu_mutex == 0) {
				this->local_process = false;
			}
		}
	}

	__device__ uint32 get_block_idx() {
		return blockIdx.x + blockIdx.y * gridDim.x
				+ gridDim.x * gridDim.y * blockIdx.z;
	}

	__device__ uint32 get_global_idx() {
		uint32 blockId = this->get_block_idx();

		uint32 threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
				+ (threadIdx.z * (blockDim.x * blockDim.y))
				+ (threadIdx.y * blockDim.x) + threadIdx.x;
//		printf("THREAD ID %d\n", threadId);

		return threadId;
	}

	__device__ void __gpu_sync() {
		__syncthreads();
		// only thread 0 is used for synchronization
		if (this->tid_in_block == 0) {
			atomicAdd(UINTCAST(&gpu_mutex), 1);
			//only when all blocks add 1 to g_mutex will
			//g_mutex equal to blocks_to_synch
			while (gpu_mutex < this->blocks_to_synch)
				;
		}
		__syncthreads();
	}

	__device__ void iteration_finished() {
		this->local_process = true;
		this->__gpu_sync();
	}

	__device__ bool keep_working() {
		return running == 1;
	}

};

#endif /* PERSISTENT_LIB_H_ */
