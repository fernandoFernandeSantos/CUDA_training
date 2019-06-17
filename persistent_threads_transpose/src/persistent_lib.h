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

typedef unsigned int uint32;
typedef unsigned char byte;

volatile __device__ byte running;
volatile __device__ uint32 gpu_mutex;

__global__ void set_gpu_mutex(const uint32 value) {
	gpu_mutex = value;
}

__global__ void set_gpu_running(const byte value) {
	running = value;
}

struct HostPersistentControler {
	cudaStream_t st;
	uint32 block_number;

	HostPersistentControler(dim3 grid_dim) :
			block_number(grid_dim.x * grid_dim.y * grid_dim.z) {
		checkCudaErrors(
				cudaStreamCreateWithFlags(&this->st, cudaStreamNonBlocking));

		this->set_running(1);

	}

	virtual ~HostPersistentControler() {
		checkCudaErrors(cudaStreamDestroy(this->st));
	}

	void sync_stream(){
		checkCudaErrors(cudaStreamSynchronize(this->st));
	}

	void end_kernel() {
		this->set_running(0);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void start_processing() {
		set_gpu_mutex<<<1, 1, 0, this->st>>>(0);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaStreamSynchronize(this->st));
	}

	void wait_gpu() {
		while (true) {
			uint32 counter;
			checkCudaErrors(
					cudaMemcpyFromSymbolAsync(&counter, gpu_mutex,
							sizeof(uint32), 0, cudaMemcpyDeviceToHost,
							this->st));
			checkCudaErrors(cudaStreamSynchronize(this->st));

			std::cout << "FINISHED " << counter << " " << this->block_number
					<< std::endl;

			if (this->block_number <= counter) {
				return;
			}
			sleep(1);
		}
	}

private:
	void set_running(byte value) {
		std::cout << "Setting running to " << bool(value) << std::endl;
		set_gpu_running<<<1, 1, 0, this->st>>>(value);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaStreamSynchronize(this->st));

		std::cout << "Running set to " << bool(value) << std::endl;
	}
};

struct PersistentKernel {
	uint32 blocks_to_synch;
	bool local_process;
	uint32 tid_in_block;

	__device__ PersistentKernel()	{
		//thread ID in a block
		this->tid_in_block = threadIdx.x + threadIdx.y + threadIdx.z;
		this->blocks_to_synch = gridDim.x * gridDim.y * gridDim.z;
		this->local_process = false;
	}

	__device__ void wait_for_work() {
		__syncthreads();
		if (this->tid_in_block == 0) {
			while (this->local_process) {
				if (gpu_mutex == 0) {
					this->local_process = false;
				}
			}
		}
		__syncthreads();
	}

	__device__ void iteration_finished() {
		this->local_process = true;
		__syncthreads();

		// only thread 0 is used for synchronization
		if (this->tid_in_block == 0) {
			atomicAdd(UINTCAST(&gpu_mutex), 1);
			//only when all blocks add 1 to g_mutex will
			//g_mutex equal to blocks_to_synch
			while (gpu_mutex < this->blocks_to_synch)
				;
//			printf("GPU MUTEX %d %d %d\n", gpu_mutex, threadIdx.x, threadIdx.y );

		}
		__syncthreads();
	}

	__device__ bool keep_working() {
		return bool(running);
	}

};

#endif /* PERSISTENT_LIB_H_ */
