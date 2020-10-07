#include <thread>
#include <iostream>
#include <vector>
#include <thread>
#include <functional>
#include <algorithm>
#include <random>

#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1

#include <cublas.h>
#include "cublas_v2.h"
#include <cuda_fp16.h>

#include "DeviceVector.h"

#include "cuda_utils.h"

#include <iostream>

#define MAIN_TYPE half
#define ITERATIONS 1

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

template<typename type_t>
std::vector<type_t> gen_rand_vector(size_t n){
    std::vector<type_t> v(n);
    
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_real_distribution<float> dist {0.1f, 0.9f};

    for(auto& i : v){
        i  = type_t(dist(mersenne_engine));
    }

    return v;
    
}

int test_execution() {
    int n_streams = 2;
    int m = 1024;
    int n = m;
    int k = n;
    MAIN_TYPE alpha = 0.1;
    MAIN_TYPE beta = 0.0;

    std::cout << "Allocating streams\n";
    StreamHandle stream_no_tensor;
    StreamHandle stream_tensor;

    std::cout << "Allocating thread array\n";

    std::vector < std::thread > thread_vector;

    std::cout << "Allocating GPU memory\n";
    


    std::vector<MAIN_TYPE> acpu = gen_rand_vector<MAIN_TYPE>(m * n);
    std::vector<MAIN_TYPE> bcpu = gen_rand_vector<MAIN_TYPE>(n * k);


    DeviceVector<MAIN_TYPE> A(acpu);
    DeviceVector<MAIN_TYPE> B(bcpu);
    DeviceVector<MAIN_TYPE> C1(m * k, 0.0f);
    DeviceVector<MAIN_TYPE> C2(m * k, 0.0f);
    
    

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
    
    //copy and compare
    int counter = 0;
    auto normal = C1.to_vector();
    auto tensor = C2.to_vector();
    for(auto i = 0; i < normal.size(); i++){
        auto ni = normal[i];
        auto ti = tensor[i];
        auto diff = std::fabs(ni - ti);
        auto relative = diff / std::fabs(ni);
        
        counter += (relative > 0.2f);
        
            //std::cout << "NOrmal " << ni << " Tensor " << ti <<  " diff " << diff << std::endl;       
    }

    std::cout << "Executing time " << mysecond() - start << std::endl;
    return counter;
}

int main(){
    
    for(auto i = 0; i < 10000; i++){
        auto ct = test_execution();
        if (ct != 0){
            std::cout << "Execution " << i << " has " << ct << " different elements that >20%\n";
        }
        
    }
}

