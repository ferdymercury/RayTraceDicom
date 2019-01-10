/**
 * \file
 * \brief float3 helper function implementations
 */
#ifndef FLOAT3_MATH_CUH
#define FLOAT3_MATH_CUH

#include "helper_float3.cuh"
#include <iostream>

CUDA_CALLABLE_MEMBER void print_float3(const float3 a)
{
    printf("%f %f %f\n", a.x, a.y, a.z);
}

CUDA_CALLABLE_MEMBER float sum_float3(const float3 a)
{
    return a.x + a.y + a.z;
}

#endif // FLOAT3_MATH_CUH
