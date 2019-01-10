#include <iostream>

#include "cuda_errchk.cuh"
#include "host_image_3d.cuh"
#include "vector_find.h"
#include "vector_interpolate.h"

int main()
{
    std::cout << double(RAY_WEIGHT_CUTOFF) << std::endl;
    return 0;
}
