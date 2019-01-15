/**
 * \file
 * \brief DensityAndSpTracerParams class declaration
 */
#ifndef DENSITY_AND_SP_TRACER_PARAMS_CUH
#define DENSITY_AND_SP_TRACER_PARAMS_CUH

#include "cuda_member.cuh"

class Float3FromFanTransform;

/**
 * \brief ...
 */
class DensityAndSpTracerParams{
public:
    /**
     * \brief Class constructor
     * \param densityScaleFact
     * \param spScaleFact
     * \param tracerSteps
     * \param fanIdxToImIdx
     */
    DensityAndSpTracerParams(const float densityScaleFact, const float spScaleFact, const unsigned int tracerSteps, const Float3FromFanTransform fanIdxToImIdx);

    /**
     * \brief ...
     * \return ...
     */
    CUDA_CALLABLE_MEMBER float getDensityScale() const;

    /**
     * \brief ...
     * \return ...
     */
    CUDA_CALLABLE_MEMBER float getSpScale() const;

    /**
     * \brief ...
     * \return ...
     */
    CUDA_CALLABLE_MEMBER unsigned int getSteps() const;

    /**
     * \brief ...
     * \param idxI
     * \param idxJ
     * \return ...
     */
    CUDA_CALLABLE_MEMBER float3 getStart(const int idxI, const int idxJ) const;

    /**
     * \brief ...
     * \param idxI
     * \param idxJ
     * \return ...
     */
    CUDA_CALLABLE_MEMBER float3 getInc(const int idxI, const int idxJ) const;

    /**
     * \brief ...
     * \param idxI
     * \param idxJ
     * \return ...
     */
    CUDA_CALLABLE_MEMBER float stepLen(const int idxI, const int idxJ) const;

private:
    float densityScale; ///< ...
    float spScale;      ///< ...
    unsigned int steps; ///< ...
    float3 coefOffset;  ///< ...
    float3 coefIdxI;    ///< ...
    float3 coefIdxJ;    ///< ...
    float3 transl;      ///< ...
    float3 corner;      ///< ...
    float3 delta;       ///< ...
    float2 dist;        ///< ...
};

#endif // DENSITY_AND_SP_TRACER_PARAMS_CUH
