for threads in 1 4 6 8
do
    for dim in 4086
    do
        OMP_NUM_THREADS=$threads ./gemm $dim
    done
done

