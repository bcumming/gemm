#pragma once

#include <chrono>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// helper for initializing cublas
// use only for demos: not threadsafe
static cublasHandle_t get_blas_handle() {
    static bool is_initialized = false;
    static cublasHandle_t cublas_handle;

    if(!is_initialized) {
        cublasCreate(&cublas_handle);
    }
    return cublas_handle;
}

void device_synchronize() {
    cudaDeviceSynchronize();
}

///////////////////////////////////////////////////////////////////////////////
// error checking
///////////////////////////////////////////////////////////////////////////////
static void check_status(cudaError_t status) {
    if(status != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
                  << cudaGetErrorString(status) << std::endl;
        exit(-1);
    }
}

///////////////////////////////////////////////////////////////////////////////
// allocating memory
///////////////////////////////////////////////////////////////////////////////

// allocate space on GPU for n instances of type T
template <typename T>
T* malloc_device(size_t n) {
    void* p;
    auto status = cudaMalloc(&p, n*sizeof(T));
    check_status(status);
    return (T*)p;
}

///////////////////////////////////////////////////////////////////////////////
// copying memory
///////////////////////////////////////////////////////////////////////////////

// copy n*T from host to device
template <typename T>
void copy_to_device(T* from, T* to, size_t n) {
    auto status = cudaMemcpy(to, from, n*sizeof(T), cudaMemcpyHostToDevice);
    check_status(status);
}

// copy n*T from device to host
template <typename T>
void copy_to_host(T* from, T* to, size_t n) {
    auto status = cudaMemcpy(to, from, n*sizeof(T), cudaMemcpyDeviceToHost);
    check_status(status);
}

///////////////////////////////////////////////////////////////////////////////
// everything below works both with gcc and nvcc
///////////////////////////////////////////////////////////////////////////////

// read command line arguments
static size_t read_arg(int argc, char** argv, size_t index, int default_value) {
    if(argc>index) {
        try {
            auto n = std::stoi(argv[index]);
            if(n<0) {
                return default_value;
            }
            return n;
        }
        catch (std::exception e) {
            std::cout << "error : invalid argument \'" << argv[index]
                      << "\', expected a positive integer." << std::endl;
            exit(-1);
        }
    }

    return default_value;
}

template <typename T>
T* malloc_host(size_t N, T value=T()) {
    T* ptr = (T*)(malloc(N*sizeof(T)));
    std::fill(ptr, ptr+N, value);

    return ptr;
}

// aliases for types used in timing host code
using clock_type    = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double>;

// return the time in seconds since the get_time function was first called
// for demos only: not threadsafe
static double get_time() {
    static auto start_time = clock_type::now();
    return duration_type(clock_type::now()-start_time).count();
}

static void gemm(double* a, double*b, double*c,
          int m, int n, int k,
          double alpha, double beta)
{
    cublasDgemm(
            get_blas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            a, m,
            b, k,
            &beta,
            c, m
    );
}

static void gemm(float* a, float*b, float*c,
          int m, int n, int k,
          float alpha, float beta)
{
    cublasSgemm(
            get_blas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            a, m,
            b, k,
            &beta,
            c, m
    );
}

