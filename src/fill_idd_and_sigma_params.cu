/**
 * \file
 * \brief FillIddAndSigmaParams class implementation
 */
#include "fill_idd_and_sigma_params.cuh"
#include "float3_from_fan_transform.cuh"
#include "vector_functions.hpp"
/*#ifndef __CUDACC__
#define __CUDACC__
#include "math_functions.h" // needed due to a bug in clangd not recognizing sqrt errf (not in helper_math.h either the host version)
#endif*/
FillIddAndSigmaParams::FillIddAndSigmaParams(const float beamEnergyIdx, const float beamEnergyScaleFact, const float beamPeakDepth, const float beamEntrySigmaSq,
    const float rRlScaleFact, const float spotDistInRays, const unsigned int nucMemoryStep, const unsigned int firstStep, const unsigned int afterLastStep, const Float3FromFanTransform fanIdxToImIdx) :
energyIdx(beamEnergyIdx), energyScaleFact(beamEnergyScaleFact), peakDepth(beamPeakDepth),  rRlScale(rRlScaleFact), first(firstStep), afterLast(afterLastStep), entrySigmaSq(beamEntrySigmaSq), spotDist(spotDistInRays),
nucMemStep(nucMemoryStep)
{
    dist = fanIdxToImIdx.getSourceDist();
    corner = fanIdxToImIdx.getFanIdxToFan().getOffset();
    delta = fanIdxToImIdx.getFanIdxToFan().getDelta();
    //float2 sigmaSqCoefs = sigmaSqAirCoefs(peakDepth);

    // Assumes beam along negative z, otherwise change dist.x and dist.y terms to negative. Works for positive and negative delta.x and delta.y
    volConst = abs(delta.x*delta.y*delta.z)*(1.0f - corner.z/dist.x - corner.z/dist.y + (corner.z*corner.z + delta.z*delta.z/12.0f)/(dist.x*dist.y));
    volLin = abs(delta.x*delta.y*delta.z)*delta.z*(-1.0f/dist.x - 1.0f/dist.y + 2.0f*corner.z/(dist.x*dist.y));
    volSq = abs(delta.x*delta.y*delta.z)*delta.z*delta.z/(dist.x*dist.y);
}

CUDA_CALLABLE_MEMBER void FillIddAndSigmaParams::initStepAndAirDiv() {/*const unsigned int idxI, const unsigned int idxJ*/
    //float deltaX = (corner.x + idxI*delta.x) / dist.x;
    //float deltaY = (corner.y + idxJ*delta.y) / dist.y;
    //float relStepLenSq = 1.0f + deltaX*deltaX + deltaY*deltaY;
    float relStepLenSq = 1.0f;
    float2 sigmaSqCoefs = sigmaSqAirCoefs(peakDepth);
    sigmaSqAirQuad = sigmaSqCoefs.x * relStepLenSq * delta.z * delta.z;
    //float zDist = corner.z + sigmaSqCoefs.y / (2.0f*sigmaSqCoefs.x); // = z_0 - z_effNozzle = z_0 - ( -b / (2*a) ), for sigma^2 = a*z^2 + b*z + spotSize^2
    float zDist = corner.z;
    sigmaSqAirLin = 2.0f*sigmaSqCoefs.x*relStepLenSq*delta.z*zDist + sigmaSqCoefs.y*delta.z;
    //stepLength =  abs(delta.z)*sqrtf(relStepLenSq);
    stepLength = abs(delta.z);
}

CUDA_CALLABLE_MEMBER float2 FillIddAndSigmaParams::voxelWidth(const unsigned int idxK) const {
    // float distScale = 1.0f + (idxK*delta.z)/(dist.x+corner.z); ?
    ///< @todo Initial version before flipping z-axis. Equals one at idxK = 0 instead of at origin, why
    return make_float2(delta.x * (1.0f-(corner.z+idxK*delta.z)/dist.x), delta.y * (1.0f-(corner.z+idxK*delta.z)/dist.y));
}

CUDA_CALLABLE_MEMBER float FillIddAndSigmaParams::getEnergyIdx() const {return energyIdx;}

CUDA_CALLABLE_MEMBER float FillIddAndSigmaParams::getEnergyScaleFact() const {return energyScaleFact;}

CUDA_CALLABLE_MEMBER float FillIddAndSigmaParams::getPeakDepth() const {return peakDepth;}

CUDA_CALLABLE_MEMBER float FillIddAndSigmaParams::getEntrySigmaSq() const {return entrySigmaSq;}

CUDA_CALLABLE_MEMBER float FillIddAndSigmaParams::getSpotDist() const {return spotDist;}

CUDA_CALLABLE_MEMBER unsigned int FillIddAndSigmaParams::getNucMemStep() const {return nucMemStep;}

CUDA_CALLABLE_MEMBER float FillIddAndSigmaParams::getStepLength() const {return stepLength;}

CUDA_CALLABLE_MEMBER float FillIddAndSigmaParams::getSigmaSqAirLin() const {return sigmaSqAirLin;}

CUDA_CALLABLE_MEMBER float FillIddAndSigmaParams::getSigmaSqAirQuad() const {return sigmaSqAirQuad;}

CUDA_CALLABLE_MEMBER float FillIddAndSigmaParams::getRRlScale()  const {return rRlScale;}

CUDA_CALLABLE_MEMBER unsigned int FillIddAndSigmaParams::getFirstStep() const {return first;}

CUDA_CALLABLE_MEMBER unsigned int FillIddAndSigmaParams::getAfterLastStep() const {return afterLast;}

CUDA_CALLABLE_MEMBER float FillIddAndSigmaParams::stepVol(const unsigned int k) const {return volConst + k*volLin + k*k*volSq;}

CUDA_CALLABLE_MEMBER float2 FillIddAndSigmaParams::sigmaSqAirCoefs(const float r0)
{
#ifndef NO_NOZZLE
    // Coefficients from calcSigmaInAir.m (note change of sign of b to compensate for beam along negative z)
    // a given in position x, b in position y
    return make_float2(0.00270f / (r0 - 4.50f), -4.39f / (r0 - 3.86f));
#else // NO_NOZZLE
    return make_float2(0.0f, 0.0f);
#endif // NO_NOZZLE
}
