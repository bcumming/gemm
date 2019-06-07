#!/bin/bash
# Run the benchmark for a range of matrix sizes

# 256 512 1024 2048 4096 8192

let "n = 128"
for ((i=1; i<=7; i++))
do
    let "n = 2 * n"
    ./gemm_hip "$n"
done


