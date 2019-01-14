/**
 * \file
 * \brief TransferParamStructDiv3 class implementation
 */
#include "transfer_param_struct_div3.cuh"
#include "float3_to_fan_transform.cuh"
#include "helper_math.h"

TransferParamStructDiv3::TransferParamStructDiv3(const Float3ToFanTransform imIdxToFanIdx)
{
    Matrix3x3 tTransp = imIdxToFanIdx.getImIdxToGantry().getMatrix().transpose();
    float3 delta = imIdxToFanIdx.getFanToFanIdx().getDelta();
    coefIdxI = tTransp.r0*delta;
    coefIdxJ = tTransp.r1*delta;
    coefOffset = imIdxToFanIdx.getImIdxToGantry().getOffset()*delta;
    globalOffset = imIdxToFanIdx.getFanToFanIdx().getOffset();
    inc = tTransp.r2*delta;
    start = make_float3(0.0f);
    normDist = make_float2(delta.z*imIdxToFanIdx.getSourceDist().x, delta.z*imIdxToFanIdx.getSourceDist().y);
}

CUDA_CALLABLE_MEMBER void TransferParamStructDiv3::init(const int idxI, const int idxJ)
{
    start = float(idxI)*coefIdxI + float(idxJ)*coefIdxJ + coefOffset;
}

CUDA_CALLABLE_MEMBER float3 TransferParamStructDiv3::getFanIdx(const int idxK) const
{
    float3 result = start+float(idxK)*inc;
    result.x *= 1 + result.z / (normDist.x-result.z); // Equals dividing by 1 - z_gantry/sourceDist_x
    result.y *= 1 + result.z / (normDist.y-result.z);
    result += globalOffset;
    return result;
}
