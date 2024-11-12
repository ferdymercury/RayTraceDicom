/**
 * \file
 * \brief TracerParamStructDiv3 class declaration
 */
#ifndef TRACER_PARAM_STRUCT_DIV3_CUH
#define TRACER_PARAM_STRUCT_DIV3_CUH

#include "cuda_member.cuh"
#include "vector_types.h"

class Float3FromFanTransform;

/**
 * \brief 3D tracing parameter struct
 */
class TracerParamStructDiv3 {
public:
    /**
     * \brief Class constructor
     * \param fanIdxToImIdx fan index to imaging index transform
     */
    TracerParamStructDiv3(const Float3FromFanTransform fanIdxToImIdx);

    //
    // brief
    // param fanIdxToFan
    // param sourceDist
    // param gantryToWorldIdx
    //
    //TracerParamStructDiv3(const Float3IdxTransform fanIdxToFan, const float sourceDist, const Float3AffineTransform gantryToWorldIdx);

    /**
     * \brief Get starting point
     * \param idxI index I
     * \param idxJ index J
     * \return starting point as float3 vector
     */
    CUDA_CALLABLE_MEMBER float3 getStart(const int idxI, const int idxJ) const;

    /**
     * \brief Get increment
     * \param idxI index I
     * \param idxJ index J
     * \return 3D increment as float3
     */
    CUDA_CALLABLE_MEMBER float3 getInc(const int idxI, const int idxJ) const;

    /**
     * \brief Get delta in Z
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getDeltaZ() const;

    /**
     * \brief Get minimum distance
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getMinDist() const;

    /**
     * \brief Get volume per distance
     * \param k index
     * \return float
     */
    CUDA_CALLABLE_MEMBER float volPerDist(const int k) const;

private:
    float3 coefOffset;  ///< 3D offset coefficient
    float3 coefIdxI;    ///< 3D index I
    float3 coefIdxJ;    ///< 3D index J
    float3 transl;      ///< 3D translation
    float minDist;      ///< minimum distance
    float deltaZ;       ///< deltaZ
    float dist;         ///< distance
    float volConst;     ///< p0 in parabola describing dV/dz
    float volLin;       ///< p1 in parabola describing dV/dz
    float volSq;        ///< p2 in parabola describing dV/dz
};

#endif //TRACER_PARAM_STRUCT_DIV3_CUH
