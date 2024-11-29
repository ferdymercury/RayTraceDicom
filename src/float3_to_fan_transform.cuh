/**
 * \file
 * \brief Float3ToFanTransform class declaration
 */
#ifndef FLOAT3_TO_FAN_TRANSFORM_CUH
#define FLOAT3_TO_FAN_TRANSFORM_CUH

#include "cuda_member.cuh"
#include "float3_affine_transform.cuh"
#include "float3_idx_transform.cuh"

class Float3FromFanTransform;

/**
 * \brief Transform to fan class
 */
class Float3ToFanTransform {
public:
    /**
     * \brief Class constructor
     * \param imIdxToGantry image index to gantry transform
     * \param sourceDist 3D source distance
     * \param fanToFanIdx fan to fan index transform
     */
    Float3ToFanTransform(const Float3AffineTransform imIdxToGantry, const float2 sourceDist, const Float3IdxTransform fanToFanIdx);

    /**
     * \brief Calculate inverse of this instance
     * \return The inverse transform
     */
    Float3FromFanTransform inverse() const;


    /**
     * \brief Return the point resulting from applying the transform to an input point
     * \param imIdx the input point
     * \return the transformed point as float3
     */
    CUDA_CALLABLE_MEMBER float3 transformPoint(const float3 imIdx) const;

    /**
     * \brief Get fanToFanIdx
     * \return the transform
     */
    Float3IdxTransform getFanToFanIdx() const;

    /**
     * \brief Get imIdxToGantry
     * \return the affine transform
     */
    Float3AffineTransform getImIdxToGantry() const;

    /**
     * \brief Get the source distance x,y
     * \return 2D distance
     */
    float2 getSourceDist() const;


    /**
     * \brief Set iITG.oneBasedToZeroBased to false and fTFI.oneBasedToZeroBased to true
    fTFI
     */
    void oneBasedToZeroBased();

private:
    Float3IdxTransform fTFI;    ///< fanToFanIdx
    Float3AffineTransform iITG; ///< imIdxToGantry
    const float2 dist;          ///< 2D source distance
};

#endif // FLOAT3_TO_FAN_TRANSFORM_CUH
