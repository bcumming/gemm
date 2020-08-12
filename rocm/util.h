#pragma once

#include <chrono>
#include <cmath>

// TODO: include file
#include <rocblas.h>

// helper for initializing cublas
// use only for demos: not threadsafe
static rocblas_handle get_rocblas_handle() {
    static bool is_initialized = false;
    static rocblas_handle handle;

    if(!is_initialized) {
        rocblas_initialize(); // this is optional
        auto status = rocblas_create_handle(&handle);
        if (status!=rocblas_status_success) {
            std::cerr << "ERROR: unable to initialize rocblas_handle." << std::endl;
            exit(1);
        }
        is_initialized=true;
    }
    return handle;
}

///////////////////////////////////////////////////////////////////////////////
// ROCM error checking
///////////////////////////////////////////////////////////////////////////////
static void rocm_check_status(hipError_t status) {
    if(status != hipSuccess) {
        std::cerr << "error: HIP API call : "
                  << hipGetErrorString(status) << std::endl;
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
    auto status = hipMalloc(&p, n*sizeof(T));
    rocm_check_status(status);
    return (T*)p;
}

///////////////////////////////////////////////////////////////////////////////
// copying memory
///////////////////////////////////////////////////////////////////////////////

// copy n*T from host to device
template <typename T>
void copy_to_device(T* from, T* to, size_t n) {
    auto status = hipMemcpy(to, from, n*sizeof(T), hipMemcpyHostToDevice);
    rocm_check_status(status);
}

// copy n*T from device to host
template <typename T>
void copy_to_host(T* from, T* to, size_t n) {
    auto status = hipMemcpy(to, from, n*sizeof(T), hipMemcpyDeviceToHost);
    rocm_check_status(status);
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

