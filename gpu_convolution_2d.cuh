/**
 * \file
 * \brief GPU 2D Convolution function declarations
 */
#ifndef GPU_CONVOLUTION_2D_CUH
#define GPU_CONVOLUTION_2D_CUH

#include "vector_types.h"

/**
 * \brief 2D gather-based Gaussian convolution in X on GPU
 * \param in pointer to 2D input data array (not owned), linearized
 * \param out pointer to 2D output data array (linearized) where convolution result will be stored (not owned, preallocated by user)
 * \param sigma Gaussian convolution sigmas in x and y
 * \param inWidth  width of the 2D matrix "in"
 * \param outWidth width of the 2D matrix "out"
 * \param height height of the 2D matrix "in"
 * \param pixelSp pixel spacing
 * \param inOutOffset shift of the output vs input
 * \param inOutDelta shift of the output vs input in perpendicular direction
 * \return void
 */
__global__ void xConvGathResampGpu(float* const in, float* const out, float2* const sigma, const int inWidth, const int outWidth, const int height, const float pixelSp, const float inOutOffset, const float inOutDelta);

/**
 * \brief 2D gather-based Gaussian convolution in Y on GPU
 * \param in pointer to 2D input data array (not owned), linearized
 * \param out pointer to 2D output data array (linearized) where convolution result will be stored (not owned, preallocated by user)
 * \param sigma Gaussian convolution sigmas in x and y
 * \param width width of the 2D matrix "in"
 * \param inHeight height of the 2D matrix "in"
 * \param outHeight height of the 2D matrix "out"
 * \param pixelSp pixel spacing
 * \param inOutOffset shift of the output vs input
 * \param inOutDelta shift of the output vs input in perpendicular direction
 * \return void
 */
__global__ void yConvGathResampGpu(float* const in, float* const out, float2* const sigma, const int width, const int inHeight, const int outHeight, const float pixelSp, const float inOutOffset, const float inOutDelta);

/**
 * \brief 2D Gaussian convolution on GPU
 * \param in pointer to 3D input data array (not owned), linearized
 * \param interm ...
 * \param out pointer to 3D output data array (linearized) where convolution result will be stored (not owned, preallocated by user)
 * \param sigmas Gaussian convolution sigmas in x and y
 * \param inDims dimensions of 3D matrix "in"
 * \param outDims dimensions of 3D matrix "out"
 * \param spotDelta ...
 * \param spotOffset ...
 * \param rayDelta ...
 * \param rayOffset ...
 * \param pxSpMult pixel spacing multiplier for rays
 * \return void
 */
void gpuConvolution2D(float* const in, float* const interm, float* const out, float2* const sigmas, const uint3 inDims, const uint3 outDims, const float3 spotDelta, const float3 spotOffset, const float3 rayDelta, const float3 rayOffset, const float2 pxSpMult);

#endif // GPU_CONVOLUTION_2D_CUH
