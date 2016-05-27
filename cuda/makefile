gemm : gemm.cu
	nvcc -std=c++11 -O3 gemm.cu -o gemm -lcublas

clean :
	rm -f gemm
	rm -f a.out
