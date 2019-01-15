/**
 * \file
 * \brief Float3FromFanTransform class declaration
 */
#ifndef FLOAT3_FROM_FAN_TRANSFORM_CUH
#define FLOAT3_FROM_FAN_TRANSFORM_CUH

#include "cuda_member.cuh"
#include "float3_affine_transform.cuh"
#include "float3_idx_transform.cuh"

class Float3ToFanTransform;

/**
 * \brief Transform ...
 */
class Float3FromFanTransform {
public:
    /**
     * \brief Class constructor
     * \param fanIdxToFan ...
     * \param sourceDist ...
     * \param gantryToImIdx ...
     */
    Float3FromFanTransform(const Float3IdxTransform fanIdxToFan, const float2 sourceDist, const Float3AffineTransform gantryToImIdx);

    /**
     * \brief ...
     * \return ...
     */
    Float3ToFanTransform inverse() const;

    /**
     * \brief ...
     * \param shift ...
     * \return ...
     */
    Float3ToFanTransform invertAndShift(const float2 shift) const;

    /**
     * \brief ...
     * \param shift ...
     * \return ...
     */
    Float3ToFanTransform invertAndShift(const float3 shift) const;

    /**
     * \brief ...
     * \param fanIdx ...
     * \return ...
     */
    CUDA_CALLABLE_MEMBER float3 transformPoint(const float3 fanIdx) const;

    /**
     * \brief ...
     * \return ...
     */
    Float3IdxTransform getFanIdxToFan() const;

    /**
     * \brief ...
     * \return ...
     */
    Float3AffineTransform getGantryToImIdx() const;

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
    Float3IdxTransform fITF;    ///< ... fanIdxToFan
    Float3AffineTransform gTII; ///< ... gantryToImIdx
    const float2 dist;          ///< ...
};

#endif // FLOAT3_FROM_FAN_TRANSFORM_CUH
