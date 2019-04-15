# Makefile assumes hipBLAS is installed in /opt/rocm/hipblas

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
-...HIP_PATH=../../..
endif

HIPBLAS_INSTALL_DIR=/opt/rocm/hipblas
HIPBLAS_INCLUDE=$(HIPBLAS_INSTALL_DIR)/include
HIPBLAS_LIB_PATH=$(HIPBLAS_INSTALL_DIR)/lib
HIPBLAS_LIB=hipblas
HIP_INCLUDE=/opt/rocm/hip/include
LDFLAGS=-L$(HIPBLAS_LIB_PATH) -l$(HIPBLAS_LIB)
HIP_LIB_PATH=/opt/rocm/hip/lib
HIP_LIB=hip_hcc
LDFLAGS+=-L$(HIP_LIB_PATH) -l$(HIP_LIB)
HCC_LIB_PATH=/opt/rocm/hcc/lib
LDFLAGS+=-L$(HCC_LIB_PATH) -lhc_am -lmcwamp
LD=hipcc
CFLAGS=-I$(HIPBLAS_INCLUDE) -I$(HIP_INCLUDE)
CPP=hipcc
OBJ=gemm_hip.o
EXE=gemm_hip

%.o: %.cpp
		$(CPP) -std=c++11 -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
		$(LD) $(OBJ) $(LDFLAGS) -o $@ 

clean:
		rm -f $(EXE) $(OBJ)
