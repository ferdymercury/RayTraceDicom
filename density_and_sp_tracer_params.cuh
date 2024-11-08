/**
 * \file
 * \brief DensityAndSpTracerParams class declaration
 */
#ifndef DENSITY_AND_SP_TRACER_PARAMS_CUH
#define DENSITY_AND_SP_TRACER_PARAMS_CUH

#include "cuda_member.cuh"
#include "vector_types.h"

class Float3FromFanTransform;

/**
 * \brief Class grouping relevant the density and stopping power tracing parameters
 */
class DensityAndSpTracerParams{
public:
    /**
     * \brief Class constructor
     * \param densityScaleFact density scaling factor
     * \param spScaleFact stopping power scaling factor
     * \param tracerSteps number of steps  for the tracer
     * \param fanIdxToImIdx 3D affine transform of index to image index
     */
    DensityAndSpTracerParams(const float densityScaleFact, const float spScaleFact, const unsigned int tracerSteps, const Float3FromFanTransform fanIdxToImIdx);

    /**
     * \brief Get density scaling factor
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getDensityScale() const;

    /**
     * \brief Get stopping power scaling factor
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getSpScale() const;

    /**
     * \brief Get number of tracing steps
     * \return float
     */
    CUDA_CALLABLE_MEMBER unsigned int getSteps() const;

    /**
     * \brief Get starting point
     * \param idxI index in x
     * \param idxJ index in y
     * \return float3
     */
    CUDA_CALLABLE_MEMBER float3 getStart(const int idxI, const int idxJ) const;

    /**
     * \brief ...
     * \param idxI index in x
     * \param idxJ index in y
     * \return float3
     */
    CUDA_CALLABLE_MEMBER float3 getInc(const int idxI, const int idxJ) const;

    /**
     * \brief Get step length
     * \param idxI index in x
     * \param idxJ index in y
     * \return float3
     */
    CUDA_CALLABLE_MEMBER float stepLen(const int idxI, const int idxJ) const;

private:
    float densityScale; ///< density scaling factor
    float spScale;      ///< stopping power scaling factor
    unsigned int steps; ///< number of steps
    float3 coefOffset;  ///< coefficient offset
    float3 coefIdxI;    ///< 3D index i
    float3 coefIdxJ;    ///< 3D index j
    float3 transl;      ///< 3D translation
    float3 corner;      ///< 3D corner
    float3 delta;       ///< 3D delta
    float2 dist;        ///< 2D distance
};

#endif // DENSITY_AND_SP_TRACER_PARAMS_CUH
