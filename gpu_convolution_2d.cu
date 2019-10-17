/**
 * \file
 * \brief GPU 2D Convolution function implementations
 */
#include "gpu_convolution_2d.cuh"
#include <cmath>
#include "cuda_runtime.h"
#include "constants.h"

__global__ void xConvGathResampGpu(float* const in, float* const out, float2* const sigma, const int inWidth, const int outWidth, const int height, const float pixelSp, const float inOutOffset, const float inOutDelta)
{
    const int idxY = blockDim.y*blockIdx.y + threadIdx.y;
    if (idxY < height) {
        const int outIdxX = blockDim.x*blockIdx.x + threadIdx.x;
        float res = 0.0f;
        float sigmaEff = sigma[blockIdx.z].x/pixelSp;
        float rSigmaEff = rsqrtf(2.0f)/sigmaEff;
        int currentInIdxX = int( ceilf( (float(outIdxX) - (CONV_SIGMA_CUTOFF*sigmaEff+HALF) - inOutOffset) / inOutDelta) ); // Furthest input idx left of  outIdxX within 3*sigma
        float dist = currentInIdxX*inOutDelta+inOutOffset - float(outIdxX); // Distance in output between source and target
        while (dist < (CONV_SIGMA_CUTOFF*sigmaEff+HALF)){
            if (currentInIdxX >= 0 && currentInIdxX < inWidth) {
                res += HALF * ( erf((dist+HALF)*rSigmaEff) - erf((dist-HALF)*rSigmaEff) ) * in[blockIdx.z*inWidth*height + idxY*inWidth + currentInIdxX];
            }
            ++currentInIdxX;
            dist = currentInIdxX*inOutDelta+inOutOffset - float(outIdxX);
        }
        out[blockIdx.z*outWidth*height + idxY*outWidth + outIdxX] = res;
    }
}

__global__ void yConvGathResampGpu(float* const in, float* const out, float2* const sigma, const int width, const int inHeight, const int outHeight, const float pixelSp, const float inOutOffset, const float inOutDelta)
{
    const int idxX = blockDim.x*blockIdx.x + threadIdx.x;
    if (idxX < width) {
        const int outIdxY = blockDim.y*blockIdx.y + threadIdx.y;
        float res = 0.0f;
        float sigmaEff = sigma[blockIdx.z].y/pixelSp;
        float rSigmaEff = rsqrtf(2.0f)/sigmaEff;
        int currentInIdxY = int( ceilf( (float(outIdxY) - (CONV_SIGMA_CUTOFF*sigmaEff+HALF) - inOutOffset) / inOutDelta) ); // Furthest input idx left of  outIdxX within 3*sigma
        float dist = currentInIdxY*inOutDelta+inOutOffset - float(outIdxY); // Distance in output between source and target
        while (dist < (CONV_SIGMA_CUTOFF*sigmaEff+HALF)){
            if (currentInIdxY >= 0 && currentInIdxY < inHeight) {
                res += HALF * ( erf((dist+HALF)*rSigmaEff) - erf((dist-HALF)*rSigmaEff) ) * in[blockIdx.z*width*inHeight + currentInIdxY*width + idxX];
            }
            ++currentInIdxY;
            dist = currentInIdxY*inOutDelta+inOutOffset - float(outIdxY);
        }
        out[blockIdx.z*width*outHeight + outIdxY*width + idxX] = res;
        ///< @todo ... Testing, remove
        //if (blockIdx.z==20 && outIdxY==60 && idxX==32) { out[blockIdx.z*width*outHeight + outIdxY*width + idxX] = 1000000.0f; }
        //else { out[blockIdx.z*width*outHeight + outIdxY*width + idxX] = 0.0f; }
    }
}

void gpuConvolution2D(float* const in, float* const interm, float* const out, float2* const sigmas, const uint3 inDims, const uint3 outDims, const float3 spotDelta, const float3 spotOffset, const float3 rayDelta, const float3 rayOffset, const float2 pxSpMult)
{
    dim3 xConvBlock(32, 8);
    dim3 xConvGrid(outDims.x/xConvBlock.x, (inDims.y+xConvBlock.y-1)/xConvBlock.y, inDims.z);
    dim3 yConvBlock(32, 8);
    dim3 yConvGrid(outDims.x/yConvBlock.x, outDims.y/yConvBlock.y, outDims.z);
    float2 inOutDelta = make_float2( spotDelta.x / rayDelta.x, spotDelta.y / rayDelta.y); // Distance between spots in ray coordinates
    float2 inOutOffset = make_float2( (spotOffset.x-rayOffset.x) / rayDelta.x, (spotOffset.y-rayOffset.y) / rayDelta.y ); // Offset when converting from spots to rays
    xConvGathResampGpu<<<xConvGrid, xConvBlock>>>(in, interm, sigmas, inDims.x, outDims.x, inDims.y, rayDelta.x*pxSpMult.x, inOutOffset.x, inOutDelta.x);
    yConvGathResampGpu<<<yConvGrid, yConvBlock>>>(interm, out, sigmas, outDims.x, inDims.y, outDims.y,  rayDelta.y*pxSpMult.y, inOutOffset.y, inOutDelta.y);
}
