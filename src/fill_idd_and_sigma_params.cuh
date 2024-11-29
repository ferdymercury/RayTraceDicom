/**
 * \file
 * \brief FillIddAndSigmaParams class declaration
 */
#ifndef FILL_IDD_AND_SIGMA_PARAMS_CUH
#define FILL_IDD_AND_SIGMA_PARAMS_CUH

#include "cuda_member.cuh"
#include "vector_types.h"

class Float3FromFanTransform;

/**
 * \brief Class dealing with filling of integral depth dose and sigma parameters
 */
class FillIddAndSigmaParams
{
public:
    /**
     * \brief Constructor
     * \param beamEnergyIdx beam index
     * \param beamEnergyScaleFact scaling factor for the beam energy
     * \param beamPeakDepth dose maximum position
     * \param beamEntrySigmaSq entrance spatial spread (variance) of beam
     * \param rRlScaleFact radiation length scaling factor
     * \param spotDistInRays spot distance in rays
     * \param nucMemoryStep nuclear step counter
     * \param firstStep first step index
     * \param afterLastStep last step index
     * \param fanIdxToImIdx 3D transform matrix from index to image volume
     */
    FillIddAndSigmaParams(const float beamEnergyIdx, const float beamEnergyScaleFact, const float beamPeakDepth, const float beamEntrySigmaSq,
    const float rRlScaleFact, const float spotDistInRays, const unsigned int nucMemoryStep, const unsigned int firstStep, const unsigned int afterLastStep, const Float3FromFanTransform fanIdxToImIdx);

    /**
     * \brief Get voxel width
     * \param idxK depth index
     * \return float2 pixel spacing X,Y at that idX
     */
    CUDA_CALLABLE_MEMBER float2 voxelWidth(const unsigned int idxK) const;

    /**
     * \brief Get energy index
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getEnergyIdx() const;

    /**
     * \brief Get energy scaling factor
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getEnergyScaleFact() const;

    /**
     * \brief Get penetration depth
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getPeakDepth() const;

    /**
     * \brief Get spatial spread at entrance (variance)
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getEntrySigmaSq() const;

    /**
     * \brief Get spot distance in rays
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getSpotDist() const;

    /**
     * \brief Get nucMemStep
     * \return float
     */
    CUDA_CALLABLE_MEMBER unsigned int getNucMemStep() const;

    /**
     * \brief Get step length
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getStepLength() const;

    /**
     * \brief Get linear factor of spatial sigma variance
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getSigmaSqAirLin() const;

    /**
     * \brief Get quadratic factor of spatial sigma variance
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getSigmaSqAirQuad() const;

    /**
     * \brief Get radiation length scaling factor
     * \return float
     */
    CUDA_CALLABLE_MEMBER float getRRlScale() const;

    /**
     * \brief Get first
     * \return float
     */
    CUDA_CALLABLE_MEMBER unsigned int getFirstStep() const;

    /**
     * \brief Get afterLast
     * \return float
     */
    CUDA_CALLABLE_MEMBER unsigned int getAfterLastStep() const;

    /**
     * \brief Get volume of this step using parabola
     * \param k index
     * \return volume as float
     */
    CUDA_CALLABLE_MEMBER float stepVol(const unsigned int k) const;

    /**
     * \brief Sets the values of stepLength and the coeficients for the constant, linear and quadratic part of:
     * sigmaSqAir^2 = sigmaSqAirQuad * k^2 + sigmaSqAirLin * k + sigmaSqAirConst
     */
    CUDA_CALLABLE_MEMBER void initStepAndAirDiv();/*const unsigned int idxI, const unsigned int idxJ*/

    /**
     * \brief Calculate coefficients of sigma_air^2 = a*z^2 + b*z + spotSize^2 for beam along the central axis
     * \param r0 parameter of equation 0.00270f / (r0 - 4.50f), -4.39f / (r0 - 3.86f)
     * \note if NO_NOZZLE defined, return 0
     * \return float2 the variance in x and y
     */
    static __host__ __device__ float2 sigmaSqAirCoefs(const float r0);

private:
    float energyIdx;        ///< beam energy index
    float energyScaleFact;  ///< energy scaling factor
    float peakDepth;        ///< penetration depth
    float stepLength;       ///< length of step
    float sigmaSqAirLin;    ///< spatial variance in air linear parameter
    float sigmaSqAirQuad;   ///< spatial variance in air quadratic parameter
    float rRlScale;         ///< radiation length scaling factor
    unsigned int first;     ///< first step index
    unsigned int afterLast; ///< last step index
    float3 corner;          ///< corner position
    float3 delta;           ///< pixel spacing in each dimension
    float2 dist;            ///< distance of ray
    float volConst;         ///< p0 in parabola describing dV/dz
    float volLin;           ///< p1 in parabola describing dV/dz
    float volSq;            ///< p2 in parabola describing dV/dz
    float entrySigmaSq;     ///< entrance spatial spread (variance) of beam
    float spotDist;         ///< spot distance in rays
    unsigned int nucMemStep;///< step in indices for nuclear correction
};

#endif //FILL_IDD_AND_SIGMA_PARAMS_CUH
