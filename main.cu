#include <iostream>

#include "cuda_errchk.cuh"

int main()
{
    std::cout << double(RAY_WEIGHT_CUTOFF) << std::endl;
    return 0;
}
