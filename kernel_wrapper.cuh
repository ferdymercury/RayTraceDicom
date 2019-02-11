/**
 * \file
 * \brief Kernel wrapper function declarations and templates
 */
#ifndef KERNEL_WRAPPER_CUH
#define KERNEL_WRAPPER_CUH

#include "host_image_3d.cuh"
#include "beam_settings.h"
#include "energy_struct.h"
#include "transfer_param_struct_div3.cuh"
#include "density_and_sp_tracer_params.cuh"
#include "fill_idd_and_sigma_params.cuh"

const unsigned int maxSuperpR = 32; ///< Largest superposition radius in pixels
const int superpTileX = 32;         ///< Must be equal to warp size!
const int superpTileY = 8;          ///< Desktop and laptop
const int minTilesInBatch = 16;     ///< Minimum number of tiles in each KS batch

/**
 * \brief ...
 * \param val ...
 * \param multiple ...
 * \return ...
 */
int roundTo(const int val, const int multiple);

/**
 * \brief ...
 * \param in ...
 * \param out ...
 * \param inDims ...
 * \return void
 */
__global__  void extendAndPadd(float* const in, float* const out, const uint3 inDims);

/**
 * \brief ...
 * \param result ...
 * \param params ...
 * \param startIdx ...
 * \param maxZ ...
 * \param doseDims ...
 * \return void
 */
__global__  void primTransfDiv(float* const result, TransferParamStructDiv3 params, const int3 startIdx, const int maxZ, const uint3 doseDims);

/**
 * \brief ...
 * \param result ...
 * \param params ...
 * \param startIdx ...
 * \param maxZ ...
 * \param doseDims ...
 * \return void
 */
__global__  void nucTransfDiv(float* const result, const TransferParamStructDiv3 params, const int3 startIdx, const int maxZ, const uint3 doseDims);

/**
 * \brief ...
 * \param bevDensity ...
 * \param bevCumulSp ...
 * \param beamFirstInside ...
 * \param firstStepOutside ...
 * \param params ...
 * \return void
 */
__global__  void fillBevDensityAndSp(float* const bevDensity, float* const bevCumulSp, int* const beamFirstInside, int* const firstStepOutside, const DensityAndSpTracerParams params);

#ifdef NUCLEAR_CORR
/**
 * \brief Calculate sigma as described in M. Soukup, M. Fippel, and M. Alber
 * A pencil beam algorithm for intensity modulated proton therapy derived from  Monte Carlo simulations.,
 * Physics in medicine and biology, vol. 50, no. 21. pp. 5089--104, 2005.
 * \param bevDensity ...
 * \param bevCumulSp ...
 * \param bevIdd ...
 * \param bevRSigmaEff ...
 * \param rayWeights ...
 * \param bevNucIdd ...
 * \param bevNucRSigmaEff ...
 * \param nucRayWeights ...
 * \param nucIdcs ...
 * \param firstInside ...
 * \param firstOutside ...
 * \param firstPassive ...
 * \param params ...
 * \return void
 * \warning This function is a bit of a mine field, only rearrange expressions if clear that they do not (explicitly or implicitly) affect subsequent expressions etc.
 */
__global__  void fillIddAndSigma(float* const bevDensity, float* const bevCumulSp, float* const bevIdd, float* const bevRSigmaEff, float* const rayWeights, float* const bevNucIdd, float* const bevNucRSigmaEff, float* const nucRayWeights, int* const nucIdcs, int* const firstInside, int* const firstOutside, int* const firstPassive, FillIddAndSigmaParams params);
#else // NUCLEAR_CORR
/**
 * \brief Calculate sigma as described in M. Soukup, M. Fippel, and M. Alber
 * A pencil beam algorithm for intensity modulated proton therapy derived from  Monte Carlo simulations.,
 * Physics in medicine and biology, vol. 50, no. 21. pp. 5089--104, 2005.
 * \param bevDensity ...
 * \param bevCumulSp ...
 * \param bevIdd ...
 * \param bevRSigmaEff ...
 * \param rayWeights ...
 * \param firstInside ...
 * \param firstOutside ...
 * \param firstPassive ...
 * \param params ...
 * \return void
 * \warning This function is a bit of a mine field, only rearrange expressions if clear that they do not (explicitly or implicitly) affect subsequent expressions etc.
 */
__global__  void fillIddAndSigma(float* const bevDensity, float* const bevCumulSp, float* const bevIdd, float* const bevRSigmaEff, float* const rayWeights, int* const firstInside, int* const firstOutside, int* const firstPassive, const FillIddAndSigmaParams params);
#endif

/**
 * \brief ...
 * \param imVol not-owning pointer to ...
 * \param doseVol not-owning pointer to ...
 * \param beams ...
 * \param iddData ...
 * \param outStream ...
 * \return void
 */
void cudaWrapperProtons(HostPinnedImage3D<float>* const imVol, HostPinnedImage3D<float>* const doseVol, const std::vector<BeamSettings> beams, const EnergyStruct iddData, std::ostream &outStream);

/**
 * \brief Finds the largest value in each slice of devIn
 * \tparam T ...
 * \tparam blockSize must divide n and be equal to blockDim.x
 * \param devIn ...
 * \param devResult ...
 * \param n the total number of elements in each slice (i.e. dim.x*dim.y)
 * \note blockDim.y, gridDim.x and gridDim.y must all be equal to 1.
 * \note gridDim.z must be equal to the number of slices.
 * \return void
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
 * \tparam T ...
 * \tparam blockSize must divide n and be equal to blockDim.x
 * \param devIn ...
 * \param devResult ...
 * \param n the total number of elements in each slice (i.e. dim.x*dim.y)
 * \note blockDim.y and gridDim.y must both be equal to 1.
 * \note gridDim.z must be equal to the number of slices.
 * \return void
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
 * \tparam blockY ...
 * \param devIn ...
 * \param startZ ...
 * \param tilePrimRadCtrs ...
 * \param inOutIdcs ...
 * \param noTiles ...
 * \note blockDim.x must be equalt to superpTileX and blockDim.y must be an even power of 2 and divide superpTileY.
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
            int rad = min(int(KS_SIGMA_CUTOFF / (sqrtf(2.0f)*tile[0]) + 0.5f), maxSuperpR+1);
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
 * \brief ...
 * \tparam T ...
 * \param devMem ...
 * \param N ...
 * \param val ...
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
//param inDose ...
//param inRSigmaEff ...
//param outDose ...
//param zFirst ...
//return void
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
//          float erfNew = erff(rSigmaEff*0.5f);
//          float erfOld = -erfNew;
//          float erfDiffsI[rad+1];
//          for (int i=0; i<=rad; ++i)
//          {
//              erfDiffsI[i] = 0.5f*(erfNew - erfOld);
//              erfOld = erfNew;
//              erfNew = erff(rSigmaEff*(float(i)+1.5f));
//          }
//          float fct = 1.0f;
//          erfNew = erff(fct*rSigmaEff*0.5f);
//          erfOld = -erfNew;
//          float erfDiffsJ[rad+1];
//          for (int j=0; j<=rad; ++j)
//          {
//              erfDiffsJ[j] = 0.5f*(erfNew - erfOld);
//              erfOld = erfNew;
//              erfNew = erff(fct*rSigmaEff*(float(j)+1.5f));
//          }
//
//          for (int i=0; i<2*rad+1; ++i)
//          {
//              float erfDiffI = erff(1.0f*rSigmaEff*(float(rad-i)+0.5f)) - erff(1.0f*rSigmaEff*(float(rad-i)-0.5f));
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
 * \brief ...
 * \tparam rad ...
 * \param inDose ...
 * \param inRSigmaEff ...
 * \param outDose ...
 * \param inDosePitch ...
 * \param inOutIdcs ...
 * \param inOutIdxPitch ...
 * \param tileCtrs ...
 * \return void
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
        if(i>=(superpTileX+2*rad)*(superpTileY+2*rad)) printf("ASD");
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
            float erfNew = erff(rSigmaEff*0.5f);
            float erfOld = -erfNew;
            float erfDiffs[rad+1]; // Warning, changed, still works?
            for (int i=0; i<=rad; ++i)
            {
                erfDiffs[i] = 0.5f*(erfNew - erfOld);
                erfOld = erfNew;
                erfNew = erff(rSigmaEff*(float(i)+1.5f));
            }
            for (int i=0; i<2*rad+1; ++i)
            {
                for (int j=0; j<2*rad+1; ++j)
                {
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
