/**
 * \file
 * \brief Float3ToFanTransform class implementation
 */
#include "float3_to_fan_transform.cuh"
#include "float3_from_fan_transform.cuh"

Float3ToFanTransform::Float3ToFanTransform(const Float3AffineTransform imIdxToGantry, const float2 sourceDist, const Float3IdxTransform fanToFanIdx) :
iITG(imIdxToGantry), dist(sourceDist), fTFI(fanToFanIdx) {};

Float3IdxTransform Float3ToFanTransform::getFanToFanIdx() const {return fTFI;}

Float3AffineTransform Float3ToFanTransform::getImIdxToGantry() const {return iITG;}

float2 Float3ToFanTransform::getSourceDist() const {return dist;}

Float3FromFanTransform Float3ToFanTransform::inverse() const {
    return Float3FromFanTransform(fTFI.inverse(), dist, iITG.inverse());
};

void Float3ToFanTransform::oneBasedToZeroBased() {
    iITG.oneBasedToZeroBased(false);
    fTFI.oneBasedToZeroBased(true);
}

CUDA_CALLABLE_MEMBER float3 Float3ToFanTransform::transformPoint(const float3 imIdx) const {
    float3 interm = iITG.transformPoint(imIdx);
    interm.x /= 1.0f - interm.z/dist.x; // z pointing away from beam direction
    interm.y /= 1.0f - interm.z/dist.y; // z pointing away from beam direction
    return fTFI.transformPoint(interm);
};
