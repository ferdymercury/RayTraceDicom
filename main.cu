#include <iostream>

#include "cuda_errchk.cuh"
#include "host_image_3d.cuh"

int main()
{
    std::cout << double(RAY_WEIGHT_CUTOFF) << std::endl;
    return 0;
}
