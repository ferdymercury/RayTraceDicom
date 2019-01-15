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
 * \brief Transform ...
 */
class Float3ToFanTransform {
public:
    /**
     * \brief Class constructor
     * \param imIdxToGantry ...
     * \param sourceDist ...
     * \param fanToFanIdx ...
     */
    Float3ToFanTransform(const Float3AffineTransform imIdxToGantry, const float2 sourceDist, const Float3IdxTransform fanToFanIdx);

    /**
     * \brief ...
     * \return ...
     */
    Float3FromFanTransform inverse() const;

    /**
     * \brief ...
     * \param imIdx
     * \return ...
     */
    CUDA_CALLABLE_MEMBER float3 transformPoint(const float3 imIdx) const;

    /**
     * \brief ...
     * \return ...
     */
    Float3IdxTransform getFanToFanIdx() const;

    /**
     * \brief ...
     * \return ...
     */
    Float3AffineTransform getImIdxToGantry() const;

    /**
     * \brief ...
     * \return ...
     */
    float2 getSourceDist() const;

    /**
     * \brief ...
     * \return void
     */
    void oneBasedToZeroBased();

private:
    Float3IdxTransform fTFI;    ///< ... fanToFanIdx
    Float3AffineTransform iITG; ///< ... imIdxToGantry
    const float2 dist;          ///< ...
};

#endif // FLOAT3_TO_FAN_TRANSFORM_CUH
