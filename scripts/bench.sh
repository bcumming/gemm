
for threads in 1 4 12
#for threads in 1 4
do
    for dim in 1024 2048 4086 8172
    do
        OMP_NUM_THREADS=$threads numactl --cpunodebind=1 ./gemm $dim
        #OMP_NUM_THREADS=$threads ./gemm $dim | grep rate
    done
done

