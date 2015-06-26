#include <iostream>

#include <cmath>
#include <cstdio>

#include <omp.h>
#include <mkl.h>
#include <immintrin.h>

using value_type = double;
using size_type  = size_t;

void gemm(double* a, double*b, double*c,
          int m, int n, int k,
          double alpha, double beta)
{
    char trans = 'N';
    dgemm(&trans, &trans, &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
}

void gemm(float* a, float*b, float*c,
          int m, int n, int k,
          float alpha, float beta)
{
    char trans = 'N';
    sgemm(&trans, &trans, &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
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

    size_t flops_per_mul = 2 * N * M * K;
    size_t total_memory = (N*M + M*K)*sizeof(value_type);

    //std::cout << "this will take " << total_memory/(1024*1024)
    //          << "MB" << std::endl;

    // c = alpha*a*b + beta*c
    // M*N = [M*K] * [K*N] + [M*N]
    value_type* a = reinterpret_cast<value_type*>(mkl_malloc(M*K*sizeof(value_type), 64));
    value_type* b = reinterpret_cast<value_type*>(mkl_malloc(K*N*sizeof(value_type), 64));
    value_type* c = reinterpret_cast<value_type*>(mkl_malloc(M*N*sizeof(value_type), 64));

    value_type alpha{1.};
    value_type beta{1.};

    for(int i=0; i<N*K; i++)
        a[i] = value_type(1);
    for(int i=0; i<K*M; i++)
        b[i] = value_type(1);
    for(int i=0; i<N*M; i++)
        c[i] = value_type(0);

    // call once
    gemm(a, b, c, M, N, K, alpha, beta);

    double time = -dsecnd();
    for(int i=0; i<num_runs; ++i) {
        gemm(a, b, c, M, N, K, alpha, beta);
    }
    time += dsecnd();

    auto time_per_mul = time / num_runs;
    auto flops = flops_per_mul / time_per_mul;

    printf("dim %6ld %4d threads %8d Gflop/s %8.2f seconds\n", N, omp_get_max_threads(), int{std::round(flops*1e-9)}, float{time_per_mul});

    return 0;
}

