/**
 * \file
 * \brief float3 helper function declarations
 */
#ifndef HELPER_FLOAT3_CUH
#define HELPER_FLOAT3_CUH

#include "cuda_member.cuh"
#include "vector_types.h"

/**
 * \brief Prints out a float3
 * \param a the float3
 * \return void
 */
CUDA_CALLABLE_MEMBER void print_float3(const float3 a);

/**
 * \brief Calculates the sum of the three elements of a float3
 * \param a the float3
 * \return the sum as float
 */
CUDA_CALLABLE_MEMBER float sum_float3(const float3 a);

#endif // HELPER_FLOAT3_CUH
