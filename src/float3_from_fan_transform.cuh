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
 * \brief Transform from fan coordinate system
 */
class Float3FromFanTransform {
public:
    /**
     * \brief Class constructor
     * \param fanIdxToFan index to fan transform
     * \param sourceDist source distance in 2D
     * \param gantryToImIdx transform from gantry to image coordinate system
     */
    Float3FromFanTransform(const Float3IdxTransform fanIdxToFan, const float2 sourceDist, const Float3AffineTransform gantryToImIdx);

    /**
     * \brief Calculate inverse of this instance
     * \return The inverse transform
     */
    Float3ToFanTransform inverse() const;

    /**
     * \brief Invert matrix and shift in 2D
     * \param shift a 2D shift
     * \return the modified transform
     */
    Float3ToFanTransform invertAndShift(const float2 shift) const;

    /**
     * \brief Invert matrix and shift in 3D
     * \param shift a 3D shift
     * \return the modified transform
     */
    Float3ToFanTransform invertAndShift(const float3 shift) const;

    /**
     * \brief Transform a 3D point according to internal matrix
     * \param fanIdx the 3D point
     * \return the modified point
     */
    CUDA_CALLABLE_MEMBER float3 transformPoint(const float3 fanIdx) const;

    /**
     * \brief Get fanIdxToFan
     * \return the idx to fan transform
     */
    Float3IdxTransform getFanIdxToFan() const;

    /**
     * \brief Get gantryToImIdx
     * \return the gantry to image index transform
     */
    Float3AffineTransform getGantryToImIdx() const;

    /**
     * \brief Get the source distance x,y
     * \return 2D distance
     */
    float2 getSourceDist() const;

    /**
     * \brief Set fITF.oneBasedToZeroBased to false and gTII.oneBasedToZeroBased to true
     */
    void oneBasedToZeroBased();

private:
    Float3IdxTransform fITF;    ///< fanIdxToFan
    Float3AffineTransform gTII; ///< gantryToImIdx
    const float2 dist;          ///< source distance x,y
};

#endif // FLOAT3_FROM_FAN_TRANSFORM_CUH
