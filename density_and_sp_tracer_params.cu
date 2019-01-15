/**
 * \file
 * \brief DensityAndSpTracerParams class implementation
 */
#include "density_and_sp_tracer_params.cuh"
#include "float3_from_fan_transform.cuh"
#include "helper_math.h"

CUDA_CALLABLE_MEMBER float DensityAndSpTracerParams::getDensityScale() const {return densityScale;}

CUDA_CALLABLE_MEMBER float DensityAndSpTracerParams::getSpScale() const {return spScale;}

CUDA_CALLABLE_MEMBER unsigned int DensityAndSpTracerParams::getSteps() const {return steps;}

CUDA_CALLABLE_MEMBER float3 DensityAndSpTracerParams::getStart(const int idxI, const int idxJ) const {return float(idxI)*coefIdxI*(1.0f-corner.z/dist.x) + float(idxJ)*coefIdxJ*(1.0f-corner.z/dist.y) + transl;}

CUDA_CALLABLE_MEMBER float3 DensityAndSpTracerParams::getInc(const int idxI, const int idxJ) const {return (coefOffset - float(idxI)*coefIdxI/dist.x - float(idxJ)*coefIdxJ/dist.y)*delta.z;}

DensityAndSpTracerParams::DensityAndSpTracerParams(const float densityScaleFact, const float spScaleFact, const unsigned int tracerSteps, const Float3FromFanTransform fanIdxToImIdx) :
densityScale(densityScaleFact), spScale(spScaleFact), steps(tracerSteps)
{
    dist = fanIdxToImIdx.getSourceDist();
    corner = fanIdxToImIdx.getFanIdxToFan().getOffset();
    delta = fanIdxToImIdx.getFanIdxToFan().getDelta();
    Matrix3x3 tTransp = fanIdxToImIdx.getGantryToImIdx().getMatrix().transpose();
    coefOffset = tTransp.r2 - tTransp.r0*corner.x/dist.x - tTransp.r1*corner.y/dist.y;
    coefIdxI = tTransp.r0*delta.x;
    coefIdxJ = tTransp.r1*delta.y;
    transl = fanIdxToImIdx.getGantryToImIdx().getOffset() + tTransp.r2*corner.z + tTransp.r0*corner.x*(1.0f-corner.z/dist.x) + tTransp.r1*corner.y*(1.0f-corner.z/dist.y);
}

CUDA_CALLABLE_MEMBER float DensityAndSpTracerParams::stepLen(const int idxI, const int idxJ) const {
    float deltaX = (corner.x + idxI*delta.x) / dist.x;
    float deltaY = (corner.y + idxJ*delta.y) / dist.y;
    return abs(delta.z)*sqrtf(1.0f + deltaX*deltaX + deltaY*deltaY);
}
