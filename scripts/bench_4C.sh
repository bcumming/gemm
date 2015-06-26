for threads in 1 4
do
    for dim in 1024 2048 4086 8172
    do
        OMP_NUM_THREADS=$threads ./gemm $dim
    done
done

