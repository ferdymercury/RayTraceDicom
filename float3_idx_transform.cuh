/**
 * \file
 * \brief Float3IdxTransform class declaration
 */
#ifndef FLOAT3_IDX_TRANSFORM_CUH
#define FLOAT3_IDX_TRANSFORM_CUH

#include "cuda_member.cuh"

/**
 * \brief Transform ...
 */
class Float3IdxTransform {
private:
    const float3 delta; ///< The 3D delta
    float3 offset;///< The 3D offset

public:
    /**
     * \brief Default class constructor. It delta to (1,1,1) and offset to zero.
     */
    Float3IdxTransform();

    /**
     * \brief Constructor of transform
     * \param dIn the delta
     * \param oIn the offset
     */
    Float3IdxTransform(const float3 dIn, const float3 oIn);

    /**
     * \brief Gets the stored delta
     * \return float3
     */
    CUDA_CALLABLE_MEMBER float3 getDelta() const;

    /**
     * \brief Gets the stored offset
     * \return float3
     */
    CUDA_CALLABLE_MEMBER float3 getOffset() const;

    /**
     * \brief Return the point resulting from applying the transform to an input point
     * \param in the input point
     * \return the transformed point as float3
     */
    CUDA_CALLABLE_MEMBER float3 transformPoint(const float3 in) const;

    /**
     * \brief ...
     * \return an instance of the inverse transform
     */
    Float3IdxTransform inverse() const;

    /**
     * \brief Shift the offset
     * \param shift the shift to be applied to the offset
     * \return the resulting shifted transform
     */
    Float3IdxTransform shiftOffset(const float3 shift) const;

    /**
     * \brief ...
     * \return void
     */
    void oneBasedToZeroBased(const bool toIdx);

    /**
     * \brief Prints the stored delta and the stored offset
     * \return void
     */
    void print() const;
};

#endif // FLOAT3_IDX_TRANSFORM_CUH
