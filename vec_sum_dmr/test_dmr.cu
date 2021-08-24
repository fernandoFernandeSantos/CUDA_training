#include <iostream>
#include <chrono>
#include <vector>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " "  << line << std::endl;
      if (abort) exit(code);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define EPSILON 1.0e-7

template<typename real_t>
using hvector =  std::vector<real_t>;


template<typename real_t>
struct dvector{
    real_t *data;
    dvector(hvector<real_t>& v){
        cudaMalloc(&data, v.size() * sizeof(real_t));
        cudaMemcpy(data, v.data(), v.size() * sizeof(real_t), cudaMemcpyHostToDevice);
    }
    
    ~dvector(){cudaFree(data);}
    void to_vector(hvector<real_t>& v){ cudaMemcpy(v.data(),  data, v.size() * sizeof(real_t)); }    
};


/**
 * ORIGINAL
 * */
template<typename real_t>

__global__ void vector_sum(real_t* a, real_t* b, real_t* c){
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = b[i] + a[i];    
}

/**
 * DMR mixed
 * */
template<typename real_t, typename half_t>
__global__ void vector_sum_dmr(real_t* a, real_t* b, real_t* c, half_t* c_half){
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto ai = a[i];
    auto bi = b[i];
    half_t bh = half_t(bi);
    half_t ah = half_t(ai);
    
    c[i] = bi + ai;
    c_half[i] = bh + ah;    
}

/**
 * Compare overload
 **/
__device__ __forceinline__ 
bool diff(double lhs, double rhs){
    return (fabs(lhs - rhs) > EPSILON);
}

__device__ __forceinline__ 
bool diff(double lhs, float rhs){
    auto lhs_float = float(lhs);
    uint32_t ulhs = *((uint32_t*) &lhs_float);
    uint32_t urhs = *((uint32_t*) &rhs);
    
    auto diff_val = (ulhs > urhs) ? ulhs - urhs : urhs - ulhs;
    return (diff_val > 2);
}

template<typename real_t, typename half_t>
__global__ void comparator(real_t* lhs, half_t* rhs){
        auto i = threadIdx.x + blockIdx.x * blockDim.x;
        auto lhsi = lhs[i];
        auto rhsi = rhs[i];
        if(diff(lhsi, rhsi)){
            printf("Thread %d - lhs %.6e rhs %.6e\n", i, lhsi, rhsi);
        }
}

int main(){
    //time counters
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    
    //sizes
    constexpr int iterations = 100;
    constexpr size_t blocks = 8192;
    constexpr size_t threads = 1024;
    constexpr size_t size = blocks * threads;
    hvector<double> a_host(size);
    hvector<double> b_host(size);
    hvector<double> c_host(size, 0);
    
    hvector<float> c_dmr_host(size, 0);

    
    for(int i = 0; i < size; i++){
      a_host[i] = i;
      b_host[i] = 1.0/double(i * 2);  
    }
    
    dvector<double> a_dev(a_host);
    dvector<double> b_dev(b_host);
    dvector<double> c_dev(c_host);
    dvector<float> c_dmr_dev(c_dmr_host);
    dvector<double> c_dmr_full_dev(c_host);

    //Original no dmr
    auto original_time_t1 = high_resolution_clock::now();
    for(int it = 0; it < iterations; it++){
        vector_sum<<<blocks, threads>>>(a_dev.data, b_dev.data, c_dev.data);
        gpuErrchk(cudaDeviceSynchronize());
    }
    auto original_time_t2 = high_resolution_clock::now();
    
    // full DMR
    auto full_dmr_time_t1 = high_resolution_clock::now();
    for(int it = 0; it < iterations; it++){
        vector_sum<<<blocks, threads>>>(a_dev.data, b_dev.data, c_dmr_full_dev.data);
        vector_sum<<<blocks, threads>>>(a_dev.data, b_dev.data, c_dev.data);
        gpuErrchk(cudaDeviceSynchronize());
        comparator<<<blocks, threads>>>(c_dmr_full_dev.data, c_dev.data);
        gpuErrchk(cudaDeviceSynchronize());
       
    }
    auto full_dmr_time_t2 = high_resolution_clock::now();
    
    
    //Mixed dmr
    auto mixed_dmr_time_t1 = high_resolution_clock::now();
    for(int it = 0; it < iterations; it++){
        vector_sum_dmr<<<blocks, threads>>>(a_dev.data, b_dev.data, c_dev.data, c_dmr_dev.data);
        comparator<<<blocks, threads>>>(c_dev.data, c_dmr_dev.data);
        gpuErrchk(cudaDeviceSynchronize());
    }
    auto mixed_dmr_time_t2 = high_resolution_clock::now();
    
    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_original =  original_time_t2 - original_time_t1;
    duration<double, std::milli> ms_full =  full_dmr_time_t2 - full_dmr_time_t1;
    duration<double, std::milli> ms_mixed =  mixed_dmr_time_t2 - mixed_dmr_time_t1;

    
    std::cout << "ms_original: " <<  ms_original.count() << "ms\n";
    std::cout << "ms_full: " << ms_full.count() << "ms\n";
    std::cout << "ms_mixed: " << ms_mixed.count() << "ms\n";
}

