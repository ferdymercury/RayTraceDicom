/**
 * \file
 * \brief TransferParamStructDiv3 class declaration
 */
#ifndef TRANSFER_PARAM_STRUCT_DIV3_CUH
#define TRANSFER_PARAM_STRUCT_DIV3_CUH

#include "cuda_member.cuh"
#include "vector_types.h"

class Float3ToFanTransform;

/**
 * \brief Class ...
 */
class TransferParamStructDiv3 {
public:
    /**
     * \brief Class constructor
     * \param imIdxToFanIdx image index to fan index transform
     */
    TransferParamStructDiv3(const Float3ToFanTransform imIdxToFanIdx);

    /**
     * \brief Initialize starting point based on idxI and idxJ
     * \param idxI 1D index I
     * \param idxJ 1D index J
     */
    CUDA_CALLABLE_MEMBER void init(const int idxI, const int idxJ);

    /**
     * \brief Get fan index
     * \param idxK index K
     * \return 3D vector
     */
    CUDA_CALLABLE_MEMBER float3 getFanIdx(const int idxK) const;

private:
    float3 globalOffset;///< Global 3D offset
    float3 coefOffset;  ///< Coefficients 3D offset
    float3 coefIdxI;    ///< 3d index I coefficent
    float3 coefIdxJ;    ///< 3D index J coefficient
    float3 inc;         ///< 3D increment
    float3 start;       ///< 3D starting point
    float2 normDist;    ///< 2D normalized distance vector
};

#endif // TRANSFER_PARAM_STRUCT_DIV3_CUH
