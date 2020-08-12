#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <cmath>
#include <cstdio>

#ifdef GEMMROCM
    #include "util_rocm.h"
#else
    #include "util_cuda.h"
#endif

using value_type = double;
using size_type  = size_t;

int main(int argc, char** argv){
    size_t N;
    size_t M;
    size_t K;
    double duration = 2.0; // time to continue running operations for
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
    else if(argc == 5) {
        N = std::stod(argv[1]);
        M = std::stod(argv[2]);
        K = std::stod(argv[3]);
        duration = std::stod(argv[4]);
    }
    else {
        std::cout << "there are two ways to call me:" << std::endl;
        std::cout << "\"gemm N M K duration\"" << std::endl;
        std::cout << "\"gemm N M K\"" << std::endl;
        std::cout << "\"gemm N\"" << std::endl;
        exit(1);
    }

    auto handle = get_blas_handle();

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

    device_synchronize();
    auto tstart = -get_time();
    std::vector<double> times;
    //for(int i=0; i<num_runs; ++i) {
    while (tstart+get_time()<duration) {
        device_synchronize();
        auto time = -get_time();
        gemm(a, b, c, M, N, K, alpha, beta);
        device_synchronize();
        times.push_back(time+get_time());
    }

    std::sort(times.begin(), times.end());

    // drop the slowest 10% of runs
    int nruns = times.size();
    //const int to_drop = nruns>1? nruns/10+1: 0;
    const int to_drop = nruns>1? 1: 0;
    times.resize(nruns-to_drop);
    nruns = nruns-to_drop;

    const float min  = *std::min_element(times.begin(), times.end());
    const float max  = *std::max_element(times.begin(), times.end());
    const float mean = std::accumulate(times.begin(), times.end(), 0.0)/nruns;

    auto flops = [flops_per_mul] (double t) -> int {return int(std::round(flops_per_mul/t*1e-9));};

    printf("dim %6ld; runs %8d; Gflop/s [%6d %6d %6d]; seconds [%10.8f %10.8f %10.8f]\n",
            N, nruns, flops(min), flops(mean), flops(max), min, mean, max);

    return 0;
}

