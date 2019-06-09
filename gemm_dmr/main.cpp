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

struct StreamHandle {

	cudaStream_t stream;
	cublasHandle_t handle;

	StreamHandle() {
		checkFrameworkErrors(cudaStreamCreate(&stream));
		(cublasCreate(&handle));
		(cublasSetStream(handle, stream));

	}

	~StreamHandle() {
		(cudaStreamDestroy(stream));
		(cublasDestroy(handle));
	}

};

struct Parameters {
	const DeviceVector<float>& A;
	const DeviceVector<float>& B;
	DeviceVector<float>& C;
	const float* alpha;
	const float* beta;
	int m, n, k;
	const cublasMath_t math_mode;
	const cublasHandle_t& handle;

	Parameters(const DeviceVector<float>& A, const DeviceVector<float>& B,
			DeviceVector<float>& C, const float* alpha, const float* beta,
			int m, int n, int k, const cublasMath_t math_mode,
			const cublasHandle_t handle) :
			A(A), B(B), C(C), alpha(alpha), beta(beta), m(m), n(n), k(k), math_mode(
					math_mode), handle(handle) {

	}
};

void gemm_execute_float(Parameters* p) {
	std::cout << "Thread 1 started\n";

	int lda = p->m;
	int ldb = p->n;
	int ldc = p->k;

	cublasSetMathMode(p->handle, p->math_mode);

	cublasStatus_t status = cublasSgemm(p->handle, CUBLAS_OP_N, CUBLAS_OP_N, p->m, p->n,
			p->k, p->alpha, p->A.data, lda, p->B.data, ldb, p->beta, p->C.data, ldc);
}

int main() {
	int n_streams = 2;
	int m = 8192;
	int n = m;
	int k = n;
	float alpha = 0.1;
	float beta = 0.3;

	std::cout << "Allocating streams\n";
	std::vector<StreamHandle> streams(n_streams);
	std::cout << "Allocating thread array\n";

	std::vector<std::thread> thread_vector;

	std::cout << "Allocating GPU memory\n";

	DeviceVector<float> A(m * n, 2.1);
	DeviceVector<float> B(n * k, 0.4);
	DeviceVector<float> C1(m * k, 1.0);
	DeviceVector<float> C2(m * k, 0.3);

	std::cout << "Creating  parameters\n";

	Parameters p_no_tensor(A, B, C1, &alpha, &beta, m, n, k,
			CUBLAS_DEFAULT_MATH, streams[0].handle);
	Parameters p_tensor(A, B, C2, &alpha, &beta, m, n, k,
					CUBLAS_TENSOR_OP_MATH, streams[1].handle);

	std::cout << "Starting thread 1\n";

	thread_vector.push_back(
			std::thread(gemm_execute_float, &p_no_tensor));

	std::cout << "Starting thread 2\n";

	thread_vector.push_back(
			std::thread(gemm_execute_float, &p_tensor));
}

