ROCBLAS_ROOT=/opt/rocm-3.6.0/rocblas
gemm : gemm.cpp
	hipcc -std=c++11 -O3 --amdgpu-target=gfx906,gfx908 gemm.cpp -o gemm -lrocblas -I${ROCBLAS_ROOT}/include -L${ROCBLAS_ROOT}/lib -lrocblas

clean :
	rm -f gemm
	rm -f a.out
