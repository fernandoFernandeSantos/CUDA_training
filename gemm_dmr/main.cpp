#include <thread>
#include <iostream>
#include <vector>
#include <thread>

#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1

#include <cublas.h>
#include "cublas_v2.h"
#include <cuda_fp16.h>

#include "DeviceVector.h"

#include "cuda_utils.h"

#include <iostream>

#define MAIN_TYPE half
#define ITERATIONS 100

struct StreamHandle {

	cudaStream_t stream;
	cublasHandle_t handle;

	StreamHandle() {
		checkFrameworkErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		checkBlasFrameworkErrors(cublasCreate(&handle));
		checkBlasFrameworkErrors(cublasSetStream(handle, stream));

		checkFrameworkErrors(cudaPeekAtLastError());
		checkFrameworkErrors(cudaDeviceSynchronize());

	}

	~StreamHandle() {
		checkFrameworkErrors(cudaStreamDestroy(stream));
		checkBlasFrameworkErrors(cublasDestroy(handle));
		checkFrameworkErrors(cudaPeekAtLastError());
		checkFrameworkErrors(cudaDeviceSynchronize());

	}

};

template<typename T>
struct Parameters {
	const DeviceVector<T>& A;
	const DeviceVector<T>& B;
	DeviceVector<T>& C;
	const T* alpha;
	const T* beta;
	int m, n, k, id;
	const cublasMath_t math_mode;
	const cublasHandle_t handle;
	const cudaStream_t stream;

	Parameters(const DeviceVector<T>& A, const DeviceVector<T>& B,
			DeviceVector<T>& C, const T* alpha, const T* beta,
			int m, int n, int k, const cublasMath_t math_mode,
			const cublasHandle_t handle, int id, const cudaStream_t stream) :
			A(A), B(B), C(C), alpha(alpha), beta(beta), m(m), n(n), k(k), math_mode(
					math_mode), handle(handle), id(id), stream(stream) {

	}
};

void gemm_execute_float(Parameters<float>* p) {
	std::cout << "Thread " << p->id << " started\n";

	int lda = p->m;
	int ldb = p->n;
	int ldc = p->k;

	cublasSetMathMode(p->handle, p->math_mode);

	for (int i = 0; i < ITERATIONS; i++){

		checkBlasFrameworkErrors(
				cublasSgemm(p->handle, CUBLAS_OP_N, CUBLAS_OP_N, p->m, p->n,
						p->k, p->alpha, p->A.data, lda, p->B.data, ldb, p->beta,
						p->C.data, ldc));

	}

	checkFrameworkErrors(cudaPeekAtLastError());
	checkFrameworkErrors(cudaStreamSynchronize(p->stream));
	std::cout << "Thread " << p->id << " finished\n";

}

void gemm_execute_half(Parameters<half>* p) {
	std::cout << "Thread " << p->id << " started\n";

	int lda = p->m;
	int ldb = p->n;
	int ldc = p->k;

	cublasSetMathMode(p->handle, p->math_mode);

	for (int i = 0; i < ITERATIONS; i++){

		checkBlasFrameworkErrors(
				cublasHgemm(p->handle, CUBLAS_OP_N, CUBLAS_OP_N, p->m, p->n,
						p->k, p->alpha, p->A.data, lda, p->B.data, ldb, p->beta,
						p->C.data, ldc));

	}

	checkFrameworkErrors(cudaPeekAtLastError());
	checkFrameworkErrors(cudaStreamSynchronize(p->stream));
	std::cout << "Thread " << p->id << " finished\n";

}

int main() {
	int n_streams = 2;
	int m = 4096;
	int n = m;
	int k = n;
	MAIN_TYPE alpha = 0.1;
	MAIN_TYPE beta = 0.3;

	std::cout << "Allocating streams\n";
	StreamHandle stream_no_tensor;
	StreamHandle stream_tensor;

	std::cout << "Allocating thread array\n";

	std::vector < std::thread > thread_vector;

	std::cout << "Allocating GPU memory\n";

	DeviceVector<MAIN_TYPE> A(m * n, 2.1);
	DeviceVector<MAIN_TYPE> B(n * k, 0.004);
	DeviceVector<MAIN_TYPE> C1(m * k, -1.0);
	DeviceVector<MAIN_TYPE> C2(m * k, -0.3);

	std::cout << "Creating  parameters\n";

	Parameters<MAIN_TYPE> p_no_tensor(A, B, C1, &alpha, &beta, m, n, k,
			CUBLAS_DEFAULT_MATH, stream_no_tensor.handle, 1, stream_no_tensor.stream);
	Parameters<MAIN_TYPE> p_tensor(A, B, C2, &alpha, &beta, m, n, k, CUBLAS_TENSOR_OP_MATH,
			stream_tensor.handle, 2, stream_tensor.stream);

	std::cout << "Starting thread 1\n";
	double start = mysecond();

	thread_vector.push_back(std::thread(gemm_execute_half, &p_no_tensor));

	std::cout << "Starting thread 2\n";

	thread_vector.push_back(std::thread(gemm_execute_half, &p_tensor));

	std::cout << "Waiting threads\n";
	for (auto &th : thread_vector) {
		th.join();
	}

	std::cout << "Executing time " << mysecond() - start << std::endl;
	return 0;
}

