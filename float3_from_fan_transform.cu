/**
 * \file
 * \brief Float3FromFanTransform class implementation
 */
#include "float3_from_fan_transform.cuh"
#include "float3_to_fan_transform.cuh"
#include "vector_functions.hpp"

Float3FromFanTransform::Float3FromFanTransform(const Float3IdxTransform fanIdxToFan, const float2 sourceDist, const Float3AffineTransform gantryToImIdx) :
fITF(fanIdxToFan), gTII(gantryToImIdx), dist(sourceDist) {};

Float3IdxTransform Float3FromFanTransform::getFanIdxToFan() const {return fITF;}

Float3AffineTransform Float3FromFanTransform::getGantryToImIdx() const {return gTII;}

float2 Float3FromFanTransform::getSourceDist() const {return dist;}

Float3ToFanTransform Float3FromFanTransform::inverse() const {
    return Float3ToFanTransform(gTII.inverse(), dist, fITF.inverse());
};

Float3ToFanTransform Float3FromFanTransform::invertAndShift(const float2 shift) const {
    Float3IdxTransform fTFI = fITF.inverse().shiftOffset(make_float3(shift.x, shift.y, 0.0f));
    return Float3ToFanTransform(gTII.inverse(), dist, fTFI);
};

Float3ToFanTransform Float3FromFanTransform::invertAndShift(const float3 shift) const {
    Float3IdxTransform fTFI = fITF.inverse().shiftOffset(make_float3(shift.x, shift.y, shift.z));
    return Float3ToFanTransform(gTII.inverse(), dist, fTFI);
};

void Float3FromFanTransform::oneBasedToZeroBased() {
    fITF.oneBasedToZeroBased(false);
    gTII.oneBasedToZeroBased(true);
}

CUDA_CALLABLE_MEMBER float3 Float3FromFanTransform::transformPoint(const float3 fanIdx) const {
    float3 interm = fITF.transformPoint(fanIdx);
    interm.x *= 1.0f - interm.z/dist.x; // z pointing away from beam direction
    interm.y *= 1.0f - interm.z/dist.y; // z pointing away from beam direction
    return gTII.transformPoint(interm);
};
