#include <iostream>

#include <cmath>
#include <cstdio>

#include "util.h"

using value_type = double;
using size_type  = size_t;

void gemm(double* a, double*b, double*c,
          int m, int n, int k,
          double alpha, double beta)
{
    cublasDgemm(
            get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            a, m,
            b, k,
            &beta,
            c, m
    );
}

void gemm(float* a, float*b, float*c,
          int m, int n, int k,
          float alpha, float beta)
{
    cublasSgemm(
            get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            a, m,
            b, k,
            &beta,
            c, m
    );
}

int main(int argc, char** argv){
    int num_runs = 4;

    size_t N;
    size_t M;
    size_t K;
    if(argc == 2) {
        N = std::stod(argv[1]);
        M = N;
        K = N;
    }
    else if(argc == 4) {
        N = std::stod(argv[1]);
        M = std::stod(argv[2]);
        K = std::stod(argv[3]);
    }
    else {
        std::cout << "there are two ways to call me:" << std::endl;
        std::cout << "\"gemm N M K\"" << std::endl;
        std::cout << "\"gemm N\"" << std::endl;
        exit(1);
    }

    auto cublas_handle = get_cublas_handle();

    size_t flops_per_mul = 2 * N * M * K;
    size_t total_memory = (N*M + M*K)*sizeof(value_type);

    std::cout << "this will take " << total_memory/(1024*1024)
              << "MB" << std::endl;

    // c = alpha*a*b + beta*c
    // M*N = [M*K] * [K*N] + [M*N]
    auto a_host = malloc_host<value_type>(M*K, 1);
    auto b_host = malloc_host<value_type>(K*N, 1);
    auto c_host = malloc_host<value_type>(M*N, 0);

    auto a = malloc_device<value_type>(M*K);
    auto b = malloc_device<value_type>(K*N);
    auto c = malloc_device<value_type>(M*N);

    copy_to_device<value_type>(a_host, a, M*K);
    copy_to_device<value_type>(b_host, b, K*N);
    copy_to_device<value_type>(c_host, c, M*N);

    value_type alpha{1.};
    value_type beta{1.};

    // call once
    gemm(a, b, c, M, N, K, alpha, beta);

    cudaDeviceSynchronize();
    auto time = -get_time();
    for(int i=0; i<num_runs; ++i) {
        gemm(a, b, c, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();
    time += get_time();

    auto time_per_mul = time / num_runs;
    auto flops = flops_per_mul / time_per_mul;

    printf("dim %6ld %8d Gflop/s %8.2f seconds\n", N, int(std::round(flops*1e-9)), float(time_per_mul));

    return 0;
}

