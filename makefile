gemm : gemm.cpp
	icpc -std=c++11 -O3 gemm.cpp -mkl -o gemm

clean :
	rm -f gemm
	rm -f a.out
