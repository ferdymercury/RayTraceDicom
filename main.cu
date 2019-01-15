#include <iostream>

#include "cuda_errchk.cuh"
#include "host_image_3d.cuh"
#include "vector_find.h"
#include "vector_interpolate.h"
#include "tracer_param_struct_div3.cuh"
#include "tracer_param_struct3.h"
#include "transfer_param_struct_div3.cuh"
#include "cpu_convolution_1d.h"
#include "gpu_convolution_2d.cuh"

int main()
{
    std::cout << double(RAY_WEIGHT_CUTOFF) << std::endl;
    return 0;
}
