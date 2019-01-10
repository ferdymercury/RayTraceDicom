/**
 * \file
 * \brief Float3IdxTransform class implementation
 */
#include "float3_idx_transform.cuh"
#include "helper_math.h"
#include "helper_float3.cuh"

Float3IdxTransform::Float3IdxTransform() : delta(make_float3(1.0f, 1.0f, 1.0f)), offset(make_float3(0.0f, 0.0f, 0.0f)) {}

Float3IdxTransform::Float3IdxTransform(const float3 dIn, const float3 oIn) : delta(dIn), offset(oIn) {}

CUDA_CALLABLE_MEMBER float3 Float3IdxTransform::getDelta() const {return delta;}

CUDA_CALLABLE_MEMBER float3 Float3IdxTransform::getOffset() const {return offset;}

CUDA_CALLABLE_MEMBER float3 Float3IdxTransform::transformPoint(const float3 in) const {return in*delta + offset;}

Float3IdxTransform Float3IdxTransform::inverse() const {return Float3IdxTransform(make_float3(1.0f, 1.0f, 1.0f)/delta, -1.*offset/delta);}

Float3IdxTransform Float3IdxTransform::shiftOffset(const float3 shift) const {return Float3IdxTransform(delta, offset+shift);}

void Float3IdxTransform::oneBasedToZeroBased(const bool toIdx) {
    if (toIdx) offset -= make_float3(1.0f, 1.0f, 1.0f);
    else offset = offset + delta;
}

void Float3IdxTransform::print() const {
    print_float3(delta);
    print_float3(offset);
}
