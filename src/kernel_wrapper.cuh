/**
 * \file
 * \brief Kernel wrapper function declarations and templates
 */
#ifndef KERNEL_WRAPPER_CUH
#define KERNEL_WRAPPER_CUH

#include "constants.h"
#include "host_image_3d.cuh"
#include "beam_settings.h"
#include "energy_struct.h"
#include "transfer_param_struct_div3.cuh"
#include "density_and_sp_tracer_params.cuh"
#include "fill_idd_and_sigma_params.cuh"
#include "vector_functions.hpp"

#include <ostream>

#include "device_launch_parameters.h"
/*#ifndef __CUDACC__
#define __CUDACC__
#include "device_functions.h" // needed due to a bug in clangd not recognizing __synchthreads
#include "math_functions.h" // needed due to a bug in clangd not recognizing sqrt errf (not in helper_math.h either the host version)
#endif*/

const unsigned int maxSuperpR = 32; ///< Largest superposition radius in pixels
const int superpTileX = 32;         ///< Must be equal to warp size!
const int superpTileY = 8;          ///< Desktop and laptop
const int minTilesInBatch = 16;     ///< Minimum number of tiles in each KS batch

/**
 * \brief Integer Rounding function (wrt to a multiple).
 * \param val the value
 * \param multiple the rounding step
 * \return the rounded integer
 */
int roundTo(const int val, const int multiple);

#ifdef NUCLEAR_CORR
/**
 * \brief Function to extend and pad with zeros the nuclear 3D PB map to be divisible by superpTileX and superpTileY
 * \note not all nuclear PBs necesarily map to a computational PB
 * \param in input 3D array (linearized, not owned)
 * \param out output 3D array (linearized, not owned)
 * \param inDims 3D dimensions associated with the input array
 */
__global__  void extendAndPadd(float* const in, float* const out, const uint3 inDims);
#endif

/**
 * \brief Kernel transferring from fan index to dose index
 * \param result where the resulting 3D array is stored, preallocated, not owned, linearized
 * \param params the settings of the transfer kernel
 * \param startIdx the starting 3D point (indices)
 * \param maxZ the maximum index in Z
 * \param doseDims the 3D dimensions of the dose matrix
 * \param bevPrimDoseTex 3D dose texture matrix
 */
__global__  void primTransfDiv(float* const result, TransferParamStructDiv3 params, const int3 startIdx, const int maxZ, const uint3 doseDims
#if CUDART_VERSION >= 12000
, cudaTextureObject_t bevPrimDoseTex
#endif
);

#ifdef NUCLEAR_CORR
/**
 * \brief Kernel transferring from nuclear index to dose index
 * \param result where the resulting 3D array is stored, preallocated, not owned, linearized
 * \param params the settings of the transfer kernel
 * \param startIdx the starting 3D point (indices)
 * \param maxZ the maximum index in Z
 * \param doseDims the 3D dimensions of the dose matrix
 * \param bevNucDoseTex 3D matrix containing nuclear dose for each voxel xyz
 */
__global__  void nucTransfDiv(float* const result, TransferParamStructDiv3 params, const int3 startIdx, const int maxZ, const uint3 doseDims
#if CUDART_VERSION >= 12000
, cudaTextureObject_t bevNucDoseTex
#endif
);
#endif

/**
 * \brief Kernel to fill density and stopping power matrices
 * \param bevDensity Linearized 3D array (rX*rY*nSteps), not owned, of mass density at voxel centre
 * \param bevCumulSp Linearized 3D array (rX*rY*nSteps), not owned, of WEPL to far end of voxel
 * \param beamFirstInside first step where the beam reaches the phantom
 * \param firstStepOutside first step outside volume
 * \param params density and stopping power tracing parameters
 * \param imVolTex HU per voxel 3D matrix
 * \param densityTex density 1D array as function of HU
 * \param stoppingPowerTex 1D array with stopping power as function of HU number
 */
__global__  void fillBevDensityAndSp(float* const bevDensity, float* const bevCumulSp, int* const beamFirstInside, int* const firstStepOutside, const DensityAndSpTracerParams params
#if CUDART_VERSION >= 12000
, cudaTextureObject_t imVolTex, cudaTextureObject_t densityTex, cudaTextureObject_t stoppingPowerTex
#endif
);

#ifdef NUCLEAR_CORR
/**
 * \brief Calculate sigma as described in M. Soukup, M. Fippel, and M. Alber
 * A pencil beam algorithm for intensity modulated proton therapy derived from  Monte Carlo simulations.,
 * Physics in medicine and biology, vol. 50, no. 21. pp. 5089--104, 2005.
 * \param bevDensity Linearized 3D array (rX*rY*nSteps), not owned, of mass density at voxel centre
 * \param bevCumulSp Linearized 3D array (rX*rY*nSteps), not owned, of WEPL to far end of voxel
 * \param bevIdd Linearized 3D array (rX*rY*nSteps), not owned, of ray dose before kernel superposition
 * \param bevRSigmaEff Linearized 3D array (rX*rY*nSteps), not owned, of reciprocal of effective sigmas
 * \param rayWeights Linearized 3D array (rX*rY*nSpots), not owned, of ray weights
 * \param bevNucIdd Linearized 3D array (nrX*nrY*nSteps), not owned, of dose 'nuclear' rays before kernel superposition
 * \param bevNucRSigmaEff Linearized 3D array (nrX*nrY*nSteps), not owned, of reciprocal of effective nuclear sigmas
 * \param nucRayWeights Linearized 3D array (nrX*nrY*nSteps), not owned, of weights 'nuclear' rays
 * \param nucIdcs Linearized 2D array (nrX*nrY), not owned, with map of nuclear PB indices corresponding to the indices of computational PBs
 * \param firstInside Linearized 2D array (rX*rY), not owned, of step number (compared to the global entry depth) where each ray first enters the patient
 * \param firstOutside Linearized 2D array (rX*rY), not owned, of step number (compared to the global entry depth) where each ray exits the patient
 * \param firstPassive Linearized 2D array (rX*rY), not owned, of step number (compared to the global entry depth) where each ray is no longer live
 * \param params kernel filling parameters
 * \param cumulIddTex 2D matrix with the cumulative depth-dose profile as a function of depth and initial proton energy
 * \param rRadiationLengthTex 1D array with radiation length as function of density
 * \param nucWeightTex 2D matrix with the nuclear correction factor as function of cumulative stopping power and energy
 * \param nucSqSigmaTex 2D matrix with the nuclear variance? as function of cumulative stopping power and energy
 * \warning This function is a bit of a mine field, only rearrange expressions if clear that they do not (explicitly or implicitly) affect subsequent expressions etc.
 */
__global__  void fillIddAndSigma(float* const bevDensity, float* const bevCumulSp, float* const bevIdd, float* const bevRSigmaEff, float* const rayWeights, float* const bevNucIdd, float* const bevNucRSigmaEff, float* const nucRayWeights, int* const nucIdcs, int* const firstInside, int* const firstOutside, int* const firstPassive, FillIddAndSigmaParams params
#if CUDART_VERSION >= 12000
, cudaTextureObject_t cumulIddTex, cudaTextureObject_t rRadiationLengthTex, cudaTextureObject_t nucWeightTex, cudaTextureObject_t nucSqSigmaTex
#endif
);
#else // NUCLEAR_CORR
/**
 * \brief Calculate sigma as described in M. Soukup, M. Fippel, and M. Alber
 * A pencil beam algorithm for intensity modulated proton therapy derived from  Monte Carlo simulations.,
 * Physics in medicine and biology, vol. 50, no. 21. pp. 5089--104, 2005.
 * \param bevDensity Linearized 3D array (rX*rY*nSteps), not owned, of mass density at voxel centre
 * \param bevCumulSp Linearized 3D array (rX*rY*nSteps), not owned, of WEPL to far end of voxel
 * \param bevIdd Linearized 3D array (rX*rY*nSteps), not owned, of ray dose before kernel superposition
 * \param bevRSigmaEff Linearized 3D array (rX*rY*nSteps), not owned, of reciprocal of effective sigmas
 * \param rayWeights Linearized 3D array (rX*rY*nSpots), not owned, of ray weights
 * \param firstInside Linearized 2D array (rX*rY), not owned, of step number (compared to the global entry depth) where each ray first enters the patient
 * \param firstOutside Linearized 2D array (rX*rY), not owned, of step number (compared to the global entry depth) where each ray exits the patient
 * \param firstPassive Linearized 2D array (rX*rY), not owned, of step number (compared to the global entry depth) where each ray is no longer live
 * \param params kernel filling parameters
 * \param cumulIddTex 2D matrix with the cumulative depth-dose profile as a function of depth and initial proton energy
 * \param rRadiationLengthTex 1D array with radiation length as function of density
 * \warning This function is a bit of a mine field, only rearrange expressions if clear that they do not (explicitly or implicitly) affect subsequent expressions etc.
 */
__global__  void fillIddAndSigma(float* const bevDensity, float* const bevCumulSp, float* const bevIdd, float* const bevRSigmaEff, float* const rayWeights, int* const firstInside, int* const firstOutside, int* const firstPassive, const FillIddAndSigmaParams params
#if CUDART_VERSION >= 12000
, cudaTextureObject_t cumulIddTex, cudaTextureObject_t rRadiationLengthTex
#endif
);
#endif

/**
 * \brief Main wrapper function calling the CUDA kernels internally
 * \param imVol non-owning pointer to the matrix where the patient CT is stored
 * \param doseVol non-owning pointer to the matrix where the calculated dose will be stored
 * \param beams the vector of irradiated beams
 * \param iddData the integral depth dose beam model
 * \param outStream stream where to write log messages
 */
void cudaWrapperProtons(HostPinnedImage3D<float>* const imVol, HostPinnedImage3D<float>* const doseVol, const std::vector<BeamSettings> beams, const EnergyStruct iddData, std::ostream &outStream);

/**
 * \brief Finds the largest value in each slice of devIn
 * \tparam T the array data type
 * \tparam blockSize must divide n and be equal to blockDim.x
 * \param devIn the input array, not owned, preallocated
 * \param devResult the array of resulting largest value per slice, not owned, preallocated
 * \param n the total number of elements in each slice (i.e. dim.x*dim.y)
 * \note blockDim.y, gridDim.x and gridDim.y must all be equal to 1.
 * \note gridDim.z must be equal to the number of slices.
 */
template <typename T, unsigned int blockSize>
__global__  void sliceMaxVar(T* const devIn, T* const devResult, const unsigned int n)
{
    extern __shared__ char threadMaxBuff[]; // Can't make it type T if using different T since extern
    T *threadMax = (T*)(threadMaxBuff);

    // One block per z-slice, each thread compares n/blockSize values before reduction
    T maxVal = devIn[n*blockIdx.z + threadIdx.x];
    for (unsigned int i=threadIdx.x+blockSize; i<n; i+=blockSize)
    {
        T testVal = devIn[n*blockIdx.z + i];
        if (testVal > maxVal) {maxVal = testVal;}
    }
    threadMax[threadIdx.x] = maxVal;
    __syncthreads();

    // Reduction
    for (unsigned int maxIdx=blockSize/2; maxIdx>0; maxIdx>>=1)
    {
        if (threadIdx.x<maxIdx && threadMax[threadIdx.x+maxIdx] > threadMax[threadIdx.x]) {
            threadMax[threadIdx.x] = threadMax[threadIdx.x+maxIdx];
        }
        __syncthreads(); // Can be left out when maxIdx >= 32 but doesn't seem to make a difference
    }

    // Write devResult
    if (threadIdx.x==0) {
        devResult[blockIdx.z] = threadMax[0];
        //printf("%4d %f\n", blockIdx.z, threadMax[0]);
    }
}

/**
 * \brief Finds the smallest value in each slice of devIn
 * \tparam T the array data type
 * \tparam blockSize must divide n and be equal to blockDim.x
 * \param devIn the input array, not owned, preallocated
 * \param devResult the array of resulting minimum value per slice, not owned, preallocated
 * \param n the total number of elements in each slice (i.e. dim.x*dim.y)
 * \note blockDim.y and gridDim.y must both be equal to 1.
 * \note gridDim.z must be equal to the number of slices.
 */
template <typename T, unsigned int blockSize>
__global__  void sliceMinVar(T* const devIn, T* const devResult, const unsigned int n)
{
    extern __shared__ char threadMinBuff[]; // Can't make it type T if using different T since extern
    T *threadMin = (T*)(threadMinBuff);

    // One block per z-slice, each thread compares n/blockSize values before reduction
    T minVal = devIn[n*blockIdx.z + threadIdx.x];
    for (unsigned int i=threadIdx.x+blockSize; i<n; i+=blockSize)
    {
        T testVal = devIn[n*blockIdx.z + i];
        if (testVal < minVal) {minVal = testVal;}
    }
    threadMin[threadIdx.x] = minVal;
    __syncthreads();

    // Reduction
    for (unsigned int maxIdx=blockSize/2; maxIdx>0; maxIdx>>=1)
    {
        if (threadIdx.x<maxIdx && threadMin[threadIdx.x+maxIdx] < threadMin[threadIdx.x]) {
            threadMin[threadIdx.x] = threadMin[threadIdx.x+maxIdx];
        }
        __syncthreads(); // Can be left out when maxIdx >= 32 but doesn't seem to make a difference
    }

    // Write devResult
    if (threadIdx.x==0) {
        devResult[blockIdx.z] = threadMin[0];
    }
}

/**
 * \brief Finds the smallest value in each tile of devIn
 * \tparam blockY block dimension in Y
 * \param devIn the input array, not owned, preallocated
 * \param startZ starting slice in Z
 * \param tilePrimRadCtrs pointer to counter arrays, to be added atomically
 * \param inOutIdcs 2D array, not owned, preallocated, mapping input to output indices
 * \param noTiles number of tiles
 * \note blockDim.x must be equal to superpTileX and blockDim.y must be an even power of 2 and divide superpTileY.
 */
template <unsigned int blockY>
__global__  void tileRadCalc(float* const devIn, const int startZ, int* const tilePrimRadCtrs, int2* const inOutIdcs, const int noTiles)
{
    __shared__ float tile[superpTileX*blockY]; // Can't make it type T if using different T since extern

    const int tileIdx = superpTileX*threadIdx.y + threadIdx.x;
    const int pitch = gridDim.x*superpTileX;
    const int inIdx = (startZ + blockIdx.z) * (gridDim.y*superpTileY*pitch) +
        (blockIdx.y*superpTileY + threadIdx.y) * pitch +
        blockIdx.x*superpTileX + threadIdx.x;

    // One block per tile, each thread compares tileY/blockDim.y values before reduction
    float minVal = devIn[inIdx];
    //for (int row = blockY+threadIdx.y; row < superpTileY; row += blockY)
    //{
    //  float testVal = devIn[inIdx + row*pitch];
    //  if (testVal < minVal) { minVal = testVal; }
    //}
    for (int i=1; i < superpTileY/blockY; ++i)
    {
        float testVal = devIn[inIdx + i*blockY*pitch];
        if (testVal < minVal) { minVal = testVal; }
    }
    tile[tileIdx] = minVal;
    __syncthreads();

    // Reduction over y
    for (int maxIdxY = blockY / 2; maxIdxY > 0; maxIdxY >>= 1)
    {
        if (threadIdx.y < maxIdxY && tile[tileIdx + maxIdxY*superpTileX] < tile[tileIdx]) {
            tile[tileIdx] = tile[tileIdx + maxIdxY*superpTileX];
        }
        __syncthreads();
    }

    // Reduction over x
    if (threadIdx.y == 0) {
        for (int maxIdxX = superpTileX / 2; maxIdxX>0; maxIdxX >>= 1)
        {
            if (threadIdx.x<maxIdxX && tile[threadIdx.x + maxIdxX] < tile[threadIdx.x]) {
                tile[threadIdx.x] = tile[threadIdx.x + maxIdxX];
            }
            __syncthreads(); // Can probably be left out since all threads should be in same warp
        }
        // Write devResult
        if (threadIdx.x == 0) {
            // Calc rad, atomically increment corresponding counter, and write back indices and pitches
            int rad = min(int(KS_SIGMA_CUTOFF / (sqrtf(2.0f)*tile[0]) + HALF), maxSuperpR+1);
            int radIdx = atomicAdd(tilePrimRadCtrs+rad, 1);
            //int tileNo = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
            //tileLists[rad*listPitch + radIdx] = tileNo;
            const int outIdx = (startZ + blockIdx.z) * ((gridDim.y*superpTileY+2*maxSuperpR)*(gridDim.x*superpTileX+2*maxSuperpR)) +
                (blockIdx.y*superpTileY) * (gridDim.x*superpTileX+2*maxSuperpR) +
                blockIdx.x*superpTileX; // threadIdx.x and threadIdx.y both == 0 and thus don't need to be added
            inOutIdcs[rad*noTiles + radIdx] = make_int2(inIdx, outIdx);
        }
    }
}

/**
 * \brief Fill device memory with all elements having the same value
 * \tparam T the underlying data type of the array
 * \param devMem the pointer to the array to be filled, preallocated, not owned
 * \param N the number of elements to fill, <= array size.
 * \param val the value to fill
 */
template<typename T>
__global__  void fillDevMem(T* const devMem, const unsigned int N, const T val)
{
    unsigned int idx = (blockDim.y*blockIdx.y + threadIdx.y) * (gridDim.x*blockDim.x) + blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int inc = gridDim.y*blockDim.y * gridDim.x*blockDim.x;
    while(idx < N)
    {
        devMem[idx] = val;
        idx += inc;
    }
}

//template <int rad>
//
//brief Handles different source distances for x and y, ~15 % slower ...
//param inDose input dose linearized 3D array, not owned, preallocated
//param inRSigmaEff the sigma of the Gaussian kernel
//param outDose resulting output dose linearized 3D array, not owned, preallocated
//param zFirst starting index in z
//*/
//note Requires blockDim.x to be equal to (or smaller than?) the warp size!
//Otherwise requires atomicAdd when incrementing tile[row+i][threadIdx.x+j]
//note blockDim.x must also be equal to superpTileX
//note blockDim.y has to divide superpTileY to avoid having __synchthreads inside branching statement
//note Dynamic array size must be sizeof(float)*(superpTileY+2*rad)*(superpTileX+2*rad)
//
//__global__  void kernelSuperposition(float *inDose, float *inRSigmaEff, float *outDose, unsigned int zFirst);
//{
//
//  extern volatile __shared__ float tile[];
//  for (int i = threadIdx.y*blockDim.x+threadIdx.x; i<(superpTileY+2*rad)*(superpTileX+2*rad); i+=blockDim.x*blockDim.y)
//  {
//      tile[i] = 0.0f;
//  }
//  __syncthreads();
//
//  // __synchtreads inside this block  requires that blockDim.y divides superpTileY
//  // to avoid branching
//  for (int row = threadIdx.y; row<superpTileY; row+=blockDim.y)
//  {
//      int idx = (blockIdx.z + zFirst)*gridDim.y*superpTileY*gridDim.x*superpTileX +
//          (blockIdx.y*superpTileY + row)*gridDim.x*superpTileX +
//          blockIdx.x*superpTileX + threadIdx.x;
//      float dose = inDose[idx];
//
//      if (__syncthreads_or(dose>0.0f))
//      {
//
//          float rSigmaEff = inRSigmaEff[idx];
//          float erfNew = erff(rSigmaEff*HALF);
//          float erfOld = -erfNew;
//          float erfDiffsI[rad+1];
//          for (int i=0; i<=rad; ++i)
//          {
//              erfDiffsI[i] = HALF*(erfNew - erfOld);
//              erfOld = erfNew;
//              erfNew = erff(rSigmaEff*(float(i)+1.5f));
//          }
//          float fct = 1.0f;
//          erfNew = erff(fct*rSigmaEff*HALF);
//          erfOld = -erfNew;
//          float erfDiffsJ[rad+1];
//          for (int j=0; j<=rad; ++j)
//          {
//              erfDiffsJ[j] = HALF*(erfNew - erfOld);
//              erfOld = erfNew;
//              erfNew = erff(fct*rSigmaEff*(float(j)+1.5f));
//          }
//
//          for (int i=0; i<2*rad+1; ++i)
//          {
//              float erfDiffI = erff(1.0f*rSigmaEff*(float(rad-i)+HALF)) - erff(1.0f*rSigmaEff*(float(rad-i)-HALF));
//              for (int j=0; j<2*rad+1; ++j)
//              {
//                  tile[(row+i)*(superpTileX+2*rad) + threadIdx.x+j] += dose*erfDiffsI[abs(rad-i)]*erfDiffsJ[abs(rad-j)];
//                  //tile[(row+i)*(superpTileX+2*rad) + threadIdx.x+j] += dose*erfDiffsI[abs(rad-i)]*erfDiffsI[abs(rad-j)];
//              }
//              __syncthreads(); // Care has to be taken to leave this out of branching statement
//          }
//      }
//  }
//
//  for (int row=threadIdx.y-rad+maxSuperpR; row<superpTileY+rad+maxSuperpR; row+=blockDim.y)
//  {
//
//      for (int col=threadIdx.x-rad+maxSuperpR; col<superpTileX+rad+maxSuperpR; col+=superpTileX)
//      {
//          int xSize = gridDim.x*superpTileX+2*maxSuperpR;
//          int zOffset = (blockIdx.z + zFirst) * (gridDim.y*superpTileY+2*maxSuperpR)*xSize;
//          int idx = zOffset + (blockIdx.y*superpTileY + row)*xSize + blockIdx.x*superpTileX + col;
//          atomicAdd(outDose+idx, tile[(row+rad-maxSuperpR)*(superpTileX+2*rad) + col+rad-maxSuperpR]);
//      }
//  }
//}

/**
 * \brief Overlay dose with Gaussian kernel in a given radius
 * \tparam rad the radius in pixels
 * \param inDose input dose linearized 3D array, not owned, preallocated
 * \param inRSigmaEff the sigma of the Gaussian kernel
 * \param outDose resulting output dose linearized 3D array, not owned, preallocated
 * \param inDosePitch input dose matrix pitch
 * \param inOutIdcs output dose matrix
 * \param inOutIdxPitch input to ouput indices mapping
 * \param tileCtrs tile counters
 * \note Requires blockDim.x to be equal to (or smaller than?) the wrap size
 * Otherwise requires atomicAdd when incrementing tile[row+i][threadIdx.x+j]
 * \note blockDim.x must also be equal to superpTileX
 * \note blockDim.y has to divide superpTileY to avoid having __synchthreads inside branching statement
 */
template <int rad>
__global__  void kernelSuperposition(float const* __restrict__ inDose, float const* __restrict__ inRSigmaEff, float* const outDose, const int inDosePitch, int2* const inOutIdcs, const int inOutIdxPitch, int* const tileCtrs)
//void kernelSuperposition(float const* __restrict__ inDose, float const* __restrict__ inRSigmaEff, float* const outDose, const int inIdx, const int outIdx, const int inPitch)
{
    volatile __shared__ float tile[(superpTileX+2*rad)*(superpTileY+2*rad)];
    for (int i = threadIdx.y*blockDim.x+threadIdx.x; i<(superpTileY+2*rad)*(superpTileX+2*rad); i+=blockDim.x*blockDim.y)
    {
        tile[i] = 0.0f;
    }
    __syncthreads();

    int tileIdx = blockIdx.x;
    int radIdx = rad;
    while (tileIdx >= tileCtrs[radIdx]) {
        tileIdx -= tileCtrs[radIdx];
        radIdx -= 1;
    }

    // __synchtreads inside this block requires that blockDim.y divides superpTileY to avoid branching
    for (int row = threadIdx.y; row<superpTileY; row+=blockDim.y)
    {
        int inIdx = inOutIdcs[radIdx*inOutIdxPitch + tileIdx].x + row*inDosePitch + threadIdx.x;
        float dose = inDose[inIdx];

        if (__syncthreads_or(dose>0.0f))
        {
            float rSigmaEff = inRSigmaEff[inIdx];
            float erfNew = erff(rSigmaEff*HALF);
            float erfOld = -erfNew;
            float erfDiffs[rad+1]; // Warning, changed, still works?
            for (int i=0; i<=rad; ++i)
            {
                erfDiffs[i] = HALF*(erfNew - erfOld);
                erfOld = erfNew;
                erfNew = erff(rSigmaEff*(float(i)+1.5f));
            }
            for (int i=0; i<2*rad+1; ++i)
            {
                for (int j=0; j<2*rad+1; ++j)
                {
                    //~ if(((row+i)*(superpTileX+2*rad) + threadIdx.x+j)>=(superpTileX+2*rad)*(superpTileY+2*rad)) printf("Dummy %d, f=%f\n", threadIdx.x, i);///<@todo fix this bug
                    tile[(row+i)*(superpTileX+2*rad) + threadIdx.x+j] += dose*erfDiffs[abs(rad-i)]*erfDiffs[abs(rad-j)];
                }
                __syncthreads(); // Care has to be taken to leave this out of branching statement
            }
        }
    }

    for (int row=threadIdx.y-rad+maxSuperpR; row<superpTileY+rad+maxSuperpR; row+=blockDim.y)
    {
        for (int col=threadIdx.x-rad+maxSuperpR; col<superpTileX+rad+maxSuperpR; col+=superpTileX)
        {
            int outPitch = inDosePitch+2*maxSuperpR;
            int outIdx = inOutIdcs[radIdx*inOutIdxPitch + tileIdx].y + row*outPitch + col;
            atomicAdd(outDose+outIdx, tile[(row+rad-maxSuperpR)*(superpTileX+2*rad) + col+rad-maxSuperpR]);
        }
    }
}

#endif // KERNEL_WRAPPER_CUH
