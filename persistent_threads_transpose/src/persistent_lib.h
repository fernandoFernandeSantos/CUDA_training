/*
 * persistent_lib.h
 *
 *  Created on: 14/06/2019
 *      Author: fernando
 */

#ifndef PERSISTENT_LIB_H_
#define PERSISTENT_LIB_H_

#define MAXTHREADNUMBER 2048

typedef unsigned long long int64;
typedef unsigned char byte;

//__device__ int64 threads_finished = 0;
__device__ bool running;
//__device__ bool process;

__device__ byte thread_flags[MAXTHREADNUMBER];

__device__ static void sleep_cuda(int64 clock_count) {
	int64 start = clock64();
	int64 clock_offset = 0;
	while (clock_offset < clock_count) {
		clock_offset = clock64() - start;
	}
}

struct HostPersistentControler {
	cudaStream_t st;
	int64 thread_number;
	std::vector<byte> host_thread_flags;
	const std::vector<byte> zero_vector;

	HostPersistentControler(int64 thread_number) :
			thread_number(thread_number), zero_vector(
					std::vector < byte > (MAXTHREADNUMBER, 0)) {
		checkCudaErrors(
				cudaStreamCreateWithFlags(&this->st, cudaStreamNonBlocking));
//		this->set_process(false);
		this->set_running(false);
		this->host_thread_flags = std::vector < byte > (MAXTHREADNUMBER, 0);
	}

	virtual ~HostPersistentControler() {
		checkCudaErrors(cudaStreamDestroy(this->st));
	}

	void start_kernel() {
		this->set_running(true);
	}

	void end_kernel() {
		this->set_running(false);
	}

	void start_process() {
//		this->set_process(true);
		this->reset_threads_finished();

	}

	void end_process() {
//		this->set_process(false);
	}

	template<typename T>
	void memcpy_from_stream(T* host_data, T* device_data, size_t size) {
		checkCudaErrors(
				cudaMemcpyAsync(host_data, device_data, size * sizeof(T),
						cudaMemcpyDeviceToHost, this->st));

		checkCudaErrors(cudaStreamSynchronize(this->st));
	}

	void wait_cuda() {
		while (true) {
			checkCudaErrors(
					cudaMemcpyFromSymbolAsync(this->host_thread_flags.data(),
							thread_flags,
							MAXTHREADNUMBER, 0, cudaMemcpyDeviceToHost, st));
			checkCudaErrors(cudaStreamSynchronize(st));
			int counter = 0;
			for (byte bt : host_thread_flags) {
				if (bt == 1)
					counter++;
			}
			std::cout << "FINISHED " << counter << " " << thread_number
					<< std::endl;

			if (thread_number <= counter) {
				return;
			}
			sleep(1);
		}
	}

	void reset_threads_finished() {
//		int64 tmp = 0;

		checkCudaErrors(
				cudaMemcpyToSymbolAsync(thread_flags, this->zero_vector.data(),
						MAXTHREADNUMBER, 0, cudaMemcpyHostToDevice, st));
		checkCudaErrors(cudaStreamSynchronize(st));

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

//	void set_process(bool value) {
//		std::cout << "Setting process " << value << std::endl;
//		checkCudaErrors(
//				cudaMemcpyToSymbolAsync(process, &value, sizeof(bool), 0,
//						cudaMemcpyHostToDevice, st));
//		checkCudaErrors(cudaStreamSynchronize(st));
//
//		std::cout << "Process set to " << value << std::endl;
//	}
};

struct PersistentKernel {
//	bool local_execute;
	bool& running_;
//	bool& process_;
//	int64& threads_finished_;

	__device__ PersistentKernel() :
			running_(running) {

//		, process_(process), threads_finished_(
//					threads_finished) {
//		this->local_execute = false;

	}

	__device__ void wait_for_work() {
		while (thread_flags[get_global_idx()] == 1)
			;
	}

	__device__ int get_global_idx() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x
				+ gridDim.x * gridDim.y * blockIdx.z;

		int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
				+ (threadIdx.z * (blockDim.x * blockDim.y))
				+ (threadIdx.y * blockDim.x) + threadIdx.x;

		return threadId;
	}

	__device__ void complete_work() {
//		if (this->process_ == false && this->threads_finished_ == 0) {
//			this->local_execute = false;
//			return;
//		}
//
//		if (this->process_ == true && this->local_execute == false) {
//			atomicAdd(&this->threads_finished_, 1);
//			this->local_execute = true;
//		}
		thread_flags[get_global_idx()] = 1;

	}

	__device__ bool stop_working() {
		return this->running_;
	}

};

#endif /* PERSISTENT_LIB_H_ */
