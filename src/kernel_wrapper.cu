/**
 * \file
 * \brief Kernel wrapper function implementations
 */
#include <iostream>
#ifdef NUCLEAR_CORR
#include <limits>
#endif
#include <ctime>

#include "helper_math.h"
#include "device_launch_parameters.h"
#include "cuda_errchk.cuh"

//#include "float3_affine_transform.cuh"
#include "float3_from_fan_transform.cuh"
#include "float3_to_fan_transform.cuh"
#include "kernel_wrapper.cuh"
#include "vector_find.h"
#include "vector_interpolate.h"
#include "gpu_convolution_2d.cuh"

// #ifndef __CUDACC__
// #define __CUDACC__ // circumvent bug in clangd
// #include "cuda_texture_types.h"
// #include "texture_fetch_functions.h"
// #include "device_functions.h"
// #endif

#if CUDART_VERSION < 12000
texture<float, cudaTextureType3D, cudaReadModeElementType> imVolTex;            ///< 3D matrix containing HU numbers + 1000 for each voxel xyz
texture<float, cudaTextureType2D, cudaReadModeElementType> cumulIddTex;         ///< 2D matrix with the cumulative depth-dose profile as a function of depth and initial proton energy
//texture<float, cudaTextureType1D, cudaReadModeElementType> peakDepthTex;        ///< 1D peak depth as function of energy?
texture<float, cudaTextureType1D, cudaReadModeElementType> densityTex;          ///< 1D array with density as function of HU number
texture<float, cudaTextureType1D, cudaReadModeElementType> stoppingPowerTex;    ///< 1D array with stopping power as function of HU number
texture<float, cudaTextureType1D, cudaReadModeElementType> rRadiationLengthTex; ///< 1D array with radiation length as function of density
texture<float, cudaTextureType3D, cudaReadModeElementType> bevPrimDoseTex;      ///< 3D matrix containing primary dose for each voxel xyz
#ifdef NUCLEAR_CORR
texture<float, cudaTextureType2D, cudaReadModeElementType> nucWeightTex;        ///< 2D matrix with the nuclear correction factor as function of cumulative stopping power and energy
texture<float, cudaTextureType2D, cudaReadModeElementType> nucSqSigmaTex;       ///< 2D matrix with the nuclear variance? as function of cumulative stopping power and energy
texture<float, cudaTextureType3D, cudaReadModeElementType> bevNucDoseTex;       ///< 3D matrix containing nuclear dose for each voxel xyz
#endif // NUCLEAR_CORR
#endif

int roundTo(const int val, const int multiple)
{
    return ((val+multiple-1)/multiple) * multiple;
}

#ifdef NUCLEAR_CORR
__global__ void extendAndPadd(float* const in, float* const out, const uint3 inDims)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockIdx.z;
    float val;
    if (x<inDims.x && y<inDims.y && z<inDims.z) {
        int inIdx = z*inDims.x*inDims.y + y*inDims.x + x;
        val = in[inIdx];
    }
    else {
        val = 0.0f;
    }
    unsigned int outIdx = z*gridDim.y*blockDim.y*gridDim.x*blockDim.x + y*gridDim.x*blockDim.x + x;
    out[outIdx] = val;
}
#endif

__global__ void primTransfDiv(float* const result, TransferParamStructDiv3 params, const int3 startIdx, const int maxZ, const uint3 doseDims
#if CUDART_VERSION >= 12000
, cudaTextureObject_t bevPrimDoseTex
#endif
)

{
    unsigned int x = startIdx.x + blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int y = startIdx.y + blockDim.y*blockIdx.y + threadIdx.y;

    if (x < doseDims.x && y < doseDims.y)
    {
        params.init(x, y); // Initiate object with current index position
        float *res = result + startIdx.z*doseDims.x*doseDims.y + y*doseDims.x + x;
        for (int z = startIdx.z; z<=maxZ; ++z) {
            float3 pos = params.getFanIdx(z) + make_float3(HALF, HALF, HALF); // Compensate for voxel value sitting at centre of voxel
            float tmp =
            #if CUDART_VERSION < 12000
            tex3D(bevPrimDoseTex, pos.x, pos.y, pos.z);
            #else
            tex3D<float>(bevPrimDoseTex, pos.x, pos.y, pos.z);
            #endif
            if (tmp > 0.0f) { // Only write to global memory if non-zero
                *res += tmp;
            }
            res += doseDims.x*doseDims.y;
        }
    }
}

#ifdef NUCLEAR_CORR
__global__ void nucTransfDiv(float* const result, const TransferParamStructDiv3 params, const int3 startIdx, const int maxZ, const uint3 doseDims
#ifdef CUDART_VERSION >= 12000
, cudaTextureObject_t bevNucDoseTex
#endif
)
{
    unsigned int x = startIdx.x + blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int y = startIdx.y + blockDim.y*blockIdx.y + threadIdx.y;

    if (x >=0 && y >= 0 && x < doseDims.x && y < doseDims.y)
    {
        params.init(x, y); // Initiate object with current index position
        float *res = result + startIdx.z*doseDims.x*doseDims.y + y*doseDims.x + x;
        for (int z = startIdx.z; z<=maxZ; ++z) {
            float3 pos = params.getFanIdx(z) + make_float3(HALF, HALF, HALF); // Compensate for voxel value sitting at centre of voxel
            float tmp =
            #ifdef CUDART_VERSION < 12000
            tex3D(bevNucDoseTex, pos.x, pos.y, pos.z);
            #else
            tex3D<float>(bevNucDoseTex, pos.x, pos.y, pos.z);
            #endif
            if (tmp > 0.0f) { // Only write to global memory if non-zero
                *res += tmp;
            }
            res += doseDims.x*doseDims.y;
        }
    }
}
#endif // NUCLEAR_CORR

__global__ void fillBevDensityAndSp(float* const bevDensity, float* const bevCumulSp, int* const beamFirstInside, int* const firstStepOutside, const DensityAndSpTracerParams params
#if CUDART_VERSION >= 12000
, cudaTextureObject_t imVolTex, cudaTextureObject_t densityTex, cudaTextureObject_t stoppingPowerTex
#endif
) {

    const unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
    const unsigned int memStep = gridDim.y*blockDim.y*gridDim.x*blockDim.x;
    unsigned int idx = y*gridDim.x*blockDim.x + x;

    // Compensate for value located at voxel corner instead of centre
    float3 pos = params.getStart(x, y) + make_float3(HALF, HALF, HALF);
    float3 step = params.getInc(x, y);
    float stepLen = params.stepLen(x, y);
    //float huPlus1000;
    float cumulSp = 0.0f;
    float cumulHuPlus1000 = 0.0f;
    int beforeFirstInside = -1;
    int lastInside = -1;

    for (unsigned int i=0; i<params.getSteps(); ++i) {
        //huPlus1000 = tex3D(imVolTex, pos.x, pos.y, pos.z) + 1000.0f;
        float huPlus1000 = 
        #if CUDART_VERSION < 12000
            tex3D(imVolTex, pos.x, pos.y, pos.z);
        #else
            tex3D<float>(imVolTex, pos.x, pos.y, pos.z);
        #endif
        cumulHuPlus1000 += huPlus1000;
        bevDensity[idx] = 
        #if CUDART_VERSION < 12000
            tex1D(densityTex, huPlus1000*params.getDensityScale() + HALF);
        #else 
            tex1Dfetch<float>(densityTex, huPlus1000*params.getDensityScale() + HALF);
        #endif

        cumulSp += 
        #if CUDART_VERSION < 12000
            stepLen * tex1D(stoppingPowerTex, huPlus1000*params.getSpScale() + HALF);
        #else
            stepLen * tex1Dfetch<float>(stoppingPowerTex, huPlus1000*params.getSpScale() + HALF);
        #endif

        if (cumulHuPlus1000 < 150.0f) {
            beforeFirstInside = i;
        }
        if (huPlus1000 > 150.0f) {
            lastInside = i;
        }
        bevCumulSp[idx] = cumulSp;

        idx += memStep;
        pos += step;
    }
    beamFirstInside[y*gridDim.x*blockDim.x + x] = beforeFirstInside+1;
    firstStepOutside[y*gridDim.x*blockDim.x + x] = lastInside+1;
}

#ifdef NUCLEAR_CORR
__global__ void fillIddAndSigma(float* const bevDensity, float* const bevCumulSp, float* const bevIdd, float* const bevRSigmaEff, float* const rayWeights, float* const bevNucIdd, float* const bevNucRSigmaEff, float* const nucRayWeights, int* const nucIdcs, int* const firstInside, int* const firstOutside, int* const firstPassive, const FillIddAndSigmaParams params
#else
__global__ void fillIddAndSigma(float* const bevDensity, float* const bevCumulSp, float* const bevIdd, float* const bevRSigmaEff, float* const rayWeights, int* const firstInside, int* const firstOutside, int* const firstPassive, FillIddAndSigmaParams params
#endif
#if CUDART_VERSION >= 12000
, cudaTextureObject_t cumulIddTex, cudaTextureObject_t rRadiationLengthTex
#ifdef NUCLEAR_CORR
, cudaTextureObject_t nucWeightTex, cudaTextureObject_t nucSqSigmaTex
#endif
#endif
) {
    const unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
    const unsigned int memStep = gridDim.y*blockDim.y*gridDim.x*blockDim.x;
    unsigned int idx = y*gridDim.x*blockDim.x + x;

    bool beamLive = true;
    const int firstIn = firstInside[idx];
    unsigned int afterLast = min(firstOutside[idx], static_cast<int>(params.getAfterLastStep())); // In case doesn't get changed
    const float rayWeight = rayWeights[idx];
    if (rayWeight < RAY_WEIGHT_CUTOFF || afterLast < params.getFirstStep()) {
        beamLive = false;
        afterLast = 0;
    }

    float res = 0.0f;
    float rSigmaEff;
    //float sigma;
    float cumulSp;
    float cumulSpOld = 0.0f;
    float cumulDose;
    float cumulDoseOld = 0.0f;
    //float stepLength = params.stepLen(x,y);
    params.initStepAndAirDiv(/*x,y*/);

    const float pInv = 0.5649718f; // 1/p, p=1.77
    const float eCoef = 8.639415f; // (10*alpha)^(-1/p), alpha=2.2e-3
    const float sqrt2 = 1.41421356f; // sqrt(2.0f)
#ifdef NUCLEAR_CORR

    // CORRECT ALL THESE

#if NUCLEAR_CORR == SOUKUP
    const float eRefSq = 190.44f; // 13.8^2, E_s^2
    const float sigmaDelta = 0.0f;
#elif NUCLEAR_CORR == FLUKA
    const float eRefSq = 216.09f; // 14.7^2, E_s^2
    const float sigmaDelta = 0.08f;
#elif NUCLEAR_CORR == GAUSS_FIT
    const float eRefSq = 169.00; // 13.0^2, E_s^2
    const float sigmaDelta = 0.06f;
#endif
#else // NUCLEAR_CORR
    const float eRefSq = 198.81f; // 14.1^2, E_s^2
    const float sigmaDelta = 0.21f;
#endif // NUCLEAR_CORR

    float incScat = 0.0f;
    float incincScat = 0.0f;
    // Value of increment when getting to params.getFirstStep()
    float incDiv = params.getSigmaSqAirLin() + (2.0f*float(params.getFirstStep()) - 1.0f) * params.getSigmaSqAirQuad();
    float sigmaSq = -incDiv; // Compensate for first addition of incDiv

#ifdef NUCLEAR_CORR
    float nucRes = 0.0f;
    float nucRSigmaEff;
    int nucIdx = nucIdcs[idx];
    float nucRayWeight;
    if (nucIdx >= 0)
    {
        nucRayWeight = nucRayWeights[nucIdx];
    }
    nucIdx += params.getFirstStep()*params.getNucMemStep();
#endif // NUCLEAR_CORR

    idx += params.getFirstStep()*memStep; // Compensate for first layer not 0
    for (unsigned int stepNo=params.getFirstStep(); stepNo<params.getAfterLastStep(); ++stepNo) {
        if (beamLive) {
            cumulSp = bevCumulSp[idx];
            cumulDose =
            #if CUDART_VERSION < 12000
                tex2D(cumulIddTex, cumulSp*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
            #else
                tex2D<float>(cumulIddTex, cumulSp*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
            #endif

            float density = bevDensity[idx]; // Consistently used throughout?
            //float peakDepth = params.getPeakDepth();

            // Sigma peaks 1 - 2 mm before the BP
            if (cumulSp < (params.getPeakDepth()))
            {
                float resE = eCoef * __powf(params.getPeakDepth() - HALF*(cumulSp+cumulSpOld), pInv); // 7.1 / 16.5 ms for __powf / powf 128x128, 512 steps on laptop
                // See Rossi et al. 1941 p. 242 for expressions for calculationg beta*p
                float betaP = resE + 938.3f - 938.3f*938.3f / (resE+938.3f); // 2.1 ms for 128x128, 512 steps on laptop
                float rRl =
                #if CUDART_VERSION < 12000
                    density * tex1D(rRadiationLengthTex, density*params.getRRlScale() + HALF);
                #else
                    density * tex1Dfetch<float>(rRadiationLengthTex, density*params.getRRlScale() + HALF);
                #endif
                float thetaSq = eRefSq/(betaP*betaP) * params.getStepLength() * rRl;

                sigmaSq += incScat + incDiv; // Adding 0.25f * thetaSq * params.getStepLength() * params.getStepLength() makes no difference
                incincScat += 2.0f * thetaSq * params.getStepLength() * params.getStepLength();
                incScat += incincScat;
                incDiv += 2.0f * params.getSigmaSqAirQuad();
            }
            else
            {
#if !defined(NUCLEAR_CORR) || NUCLEAR_CORR != GAUSS_FIT
                sigmaSq -= 1.5f * (incScat + incDiv) * density; // Empirical solution to dip in sigma after BP
#endif // !defined(NUCLEAR_CORR) || NUCLEAR_CORR != GAUSS_FIT

            }

            // Todo: Change to account for different divergence in x and y?
            rSigmaEff = HALF*(params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y) / (sqrt2 * (sqrtf(sigmaSq) + sigmaDelta)); // Empirical widening of beam
            //sigma = sqrtf(sigmaSq) + sigmaDelta;
            if (cumulSp > params.getPeakDepth()*BP_DEPTH_CUTOFF || stepNo == afterLast) {
                beamLive = false;
                afterLast = stepNo;
            }

#ifdef DOSE_TO_WATER
            float mass = (cumulSp-cumulSpOld) * params.stepVol(stepNo);
#else // DOSE_TO_WATER
            float mass = density * params.stepVol(stepNo);
#endif // DOSE_TO_WATER

#ifdef NUCLEAR_CORR
            if (mass > 1e-2f) // Avoid 0/0 and rippling effect in low density materials
            {
                float nucWeight =
                #if CUDART_VERSION < 12000
                tex2D(nucWeightTex, HALF*(cumulSp+cumulSpOld)*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
                #else
                tex2D<float>(nucWeightTex, HALF*(cumulSp+cumulSpOld)*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
                #endif
                res = (1.0f - nucWeight) * rayWeight * (cumulDose-cumulDoseOld) / mass;
                nucRes = nucWeight * nucRayWeight * (cumulDose-cumulDoseOld) / (mass*params.getSpotDist()*params.getSpotDist());
            }
            if (nucIdx >= 0)
            {
                float nucSqSigma =
                #if CUDART_VERSION < 12000
                tex2D(nucSqSigmaTex, HALF*(cumulSp+cumulSpOld)*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
                #else
                tex2D<float>(nucSqSigmaTex, HALF*(cumulSp+cumulSpOld)*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
                #endif
                nucRSigmaEff = HALF * params.getSpotDist() *(params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y) / (sqrt2 * sqrtf(sigmaSq + nucSqSigma + params.getEntrySigmaSq()));
            }
#else // NUCLEAR_CORR
            if (mass > 1e-2f) // Avoid 0/0 and ripling effect in low density materials
            {
                res = rayWeight * (cumulDose-cumulDoseOld) / mass;
            }
#endif // NUCLEAR_CORR

            cumulSpOld = cumulSp;
            cumulDoseOld = cumulDose;
        }
        if (!beamLive || static_cast<int>(stepNo)<(firstIn-1)) {
            res = 0.0f;
            rSigmaEff = __int_as_float(0x7f800000); // inf, equals sigma = 0
#ifdef NUCLEAR_CORR
            nucRes = 0.0f;
            nucRSigmaEff = __int_as_float(0x7f800000); // inf, equals sigma = 0
#endif // NUCLEAR_CORR
            //sigma = 0.0f;
        }
        bevIdd[idx] = res;
        bevRSigmaEff[idx] = rSigmaEff;
        //bevRSigmaEff[idx] = HALF*(params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y) / (sqrt2 * sigma);
        //bevRSigmaEff[idx] = 1.0f / (sqrt2 * sigma);
        //bevRSigmaEff[idx] = params.voxelWidth(stepNo).x / (sqrt2 * sigma);

#ifdef NUCLEAR_CORR
        if (nucIdx >= 0)
        {
            bevNucIdd[nucIdx] = nucRes;
            bevNucRSigmaEff[nucIdx] = nucRSigmaEff;
        }
        nucIdx += params.getNucMemStep();
#endif // NUCLEAR_CORR

        idx += memStep;
    }
    firstPassive[y*gridDim.x*blockDim.x + x] = afterLast;
}

void cudaWrapperProtons(HostPinnedImage3D<float>* const imVol, HostPinnedImage3D<float>* const doseVol, const std::vector<BeamSettings> beams, const EnergyStruct iddData, std::ostream &outStream) {


#ifdef WATER_CUBE_TEST
    outStream << "WARNING! WATER CUBE TEST ACTIVE, WATER ADDED TO RADIATION LENGTH LUT\n\n";
#endif // WATER_CUBE_TEST

    cudaErrchk(cudaFree(0)); // Initialise CUDA context

#ifdef FINE_GRAINED_TIMING
    float timeCopyAndBind = 0.0f;
    float timeAllocateAndSetup = 0.0f;
    float timeRaytracing = 0.0f;
    float timePrepareEnergyLoop = 0.0f;
    float timeFillIddSigma;
    float timePrepareSuperp;
    float timeSuperp;
    float timeCopyingToTexture = 0.0f;
    float timeTransforming = 0.0f;
    float timeCopyBack = 0.0f;
    float timeFreeMem = 0.0f;
    float timeTotal = 0.0f;
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

#else // FINE_GRAINED_TIMING
    cudaEvent_t globalStart, globalStop;
    float globalTime;
    cudaEventCreate(&globalStart);
    cudaEventCreate(&globalStop);
    cudaEventRecord(globalStart, 0);

#endif // FINE_GRAINED_TIMING

    cudaChannelFormatDesc floatChannelDesc = cudaCreateChannelDesc<float>();

    cudaArray *devImVolArr;
    const cudaExtent imExt = make_cudaExtent(imVol->getDims().x, imVol->getDims().y, imVol->getDims().z); // All sizes in elements since dealing with array
    cudaErrchk(cudaMalloc3DArray(&devImVolArr, &floatChannelDesc, imExt));
    cudaMemcpy3DParms imCopyParams = {};
    imCopyParams.srcPtr = make_cudaPitchedPtr((void*)imVol->getImData(), imExt.width*sizeof(float), imExt.width, imExt.height);
    imCopyParams.dstArray = devImVolArr;
    imCopyParams.extent = imExt;
    imCopyParams.kind = cudaMemcpyHostToDevice;
    cudaErrchk(cudaMemcpy3D(&imCopyParams));
    #if CUDART_VERSION < 12000
    imVolTex.normalized = false;
    imVolTex.filterMode = cudaFilterModeLinear;
    imVolTex.addressMode[0] = cudaAddressModeBorder;
    imVolTex.addressMode[1] = cudaAddressModeBorder;
    imVolTex.addressMode[2] = cudaAddressModeBorder;
    cudaErrchk(cudaBindTextureToArray(imVolTex, devImVolArr, floatChannelDesc));
    #else
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = devImVolArr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = imExt.width*imExt.height*imExt.depth*sizeof(float);
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    cudaTextureObject_t imVolTex=0;
    cudaCreateTextureObject(&imVolTex, &resDesc, &texDesc, NULL);
    #endif

    cudaArray *devCumulIddArr;
    cudaErrchk(cudaMallocArray(&devCumulIddArr, &floatChannelDesc, iddData.nEnergySamples, iddData.nEnergies));
    cudaErrchk(cudaMemcpyToArray(devCumulIddArr, 0, 0, &iddData.ciddMatrix[0], iddData.nEnergySamples*iddData.nEnergies*sizeof(float), cudaMemcpyHostToDevice));
    #if CUDART_VERSION < 12000
    cumulIddTex.normalized = false;
    cumulIddTex.filterMode = cudaFilterModeLinear;
    cumulIddTex.addressMode[0] = cudaAddressModeClamp;
    cumulIddTex.addressMode[1] = cudaAddressModeClamp;
    cudaErrchk(cudaBindTextureToArray(cumulIddTex, devCumulIddArr, floatChannelDesc));
    #else
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = devCumulIddArr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = iddData.nEnergySamples*iddData.nEnergies*sizeof(float);
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    cudaTextureObject_t cumulIddTex=0;
    cudaCreateTextureObject(&cumulIddTex, &resDesc, &texDesc, NULL);
    #endif

    cudaArray *devDensityArr;
    cudaErrchk(cudaMallocArray(&devDensityArr, &floatChannelDesc, iddData.nDensitySamples));
    cudaErrchk(cudaMemcpyToArray(devDensityArr, 0, 0, &iddData.densityVector[0], iddData.nDensitySamples*sizeof(float), cudaMemcpyHostToDevice));
    #if CUDART_VERSION < 12000
    densityTex.normalized = false;
    densityTex.filterMode = cudaFilterModeLinear;
    densityTex.addressMode[0] = cudaAddressModeClamp;
    cudaErrchk(cudaBindTextureToArray(densityTex, devDensityArr, floatChannelDesc));
    #else
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = devDensityArr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = iddData.nDensitySamples*sizeof(float);
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    cudaTextureObject_t densityTex=0;
    cudaCreateTextureObject(&densityTex, &resDesc, &texDesc, NULL);
    #endif
    
    cudaArray *devStoppingPowerArr;
    cudaErrchk(cudaMallocArray(&devStoppingPowerArr, &floatChannelDesc, iddData.nSpSamples));
    cudaErrchk(cudaMemcpyToArray(devStoppingPowerArr, 0, 0, &iddData.spVector[0], iddData.nSpSamples*sizeof(float), cudaMemcpyHostToDevice));
    #if CUDART_VERSION < 12000
    stoppingPowerTex.normalized = false;
    stoppingPowerTex.filterMode = cudaFilterModeLinear;
    stoppingPowerTex.addressMode[0] = cudaAddressModeClamp;
    cudaErrchk(cudaBindTextureToArray(stoppingPowerTex, devStoppingPowerArr, floatChannelDesc));
    #else
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = devStoppingPowerArr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = iddData.nSpSamples*sizeof(float);
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    cudaTextureObject_t stoppingPowerTex=0;
    cudaCreateTextureObject(&stoppingPowerTex, &resDesc, &texDesc, NULL);
    #endif

    cudaArray *devReciprocalRadiationLengthArr;
    cudaErrchk(cudaMallocArray(&devReciprocalRadiationLengthArr, &floatChannelDesc, iddData.nRRlSamples));
    cudaErrchk(cudaMemcpyToArray(devReciprocalRadiationLengthArr, 0, 0, &iddData.rRlVector[0], iddData.nRRlSamples*sizeof(float), cudaMemcpyHostToDevice));
    #if CUDART_VERSION < 12000
    rRadiationLengthTex.normalized = false;
    rRadiationLengthTex.filterMode = cudaFilterModeLinear;
    rRadiationLengthTex.addressMode[0] = cudaAddressModeClamp;
    cudaErrchk(cudaBindTextureToArray(rRadiationLengthTex, devReciprocalRadiationLengthArr, floatChannelDesc));
    #else
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = devReciprocalRadiationLengthArr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = iddData.nRRlSamples*sizeof(float);
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    cudaTextureObject_t rRadiationLengthTex=0;
    cudaCreateTextureObject(&rRadiationLengthTex, &resDesc, &texDesc, NULL);
    #endif

    float *devDoseBox;
    const int doseN = doseVol->getDims().x*doseVol->getDims().y*doseVol->getDims().z;
    cudaErrchk(cudaMalloc((void**)&devDoseBox, doseN*sizeof(float)));
    cudaErrchk(cudaMemcpy(devDoseBox, doseVol->getImData(), doseN*sizeof(float), cudaMemcpyHostToDevice));

    #ifdef NUCLEAR_CORR
    cudaArray *devNucWeightArr;
    cudaErrchk(cudaMallocArray(&devNucWeightArr, &floatChannelDesc, iddData.nEnergySamples, iddData.nEnergies));
    cudaErrchk(cudaMemcpyToArray(devNucWeightArr, 0, 0, &iddData.nucWeightMatrix[0], iddData.nEnergySamples*iddData.nEnergies*sizeof(float), cudaMemcpyHostToDevice));
    #if CUDART_VERSION < 12000
    nucWeightTex.normalized = false;
    nucWeightTex.filterMode = cudaFilterModeLinear;
    nucWeightTex.addressMode[0] = cudaAddressModeClamp;
    nucWeightTex.addressMode[1] = cudaAddressModeClamp;
    cudaErrchk(cudaBindTextureToArray(nucWeightTex, devNucWeightArr, floatChannelDesc));
    #else
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = devNucWeightArr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = iddData.nEnergySamples*iddData.nEnergies*sizeof(float);
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    cudaTextureObject_t nucWeightTex=0;
    cudaCreateTextureObject(&nucWeightTex, &resDesc, &texDesc, NULL);
    #endif

    cudaArray *devNucSqSigmaArr;
    cudaErrchk(cudaMallocArray(&devNucSqSigmaArr, &floatChannelDesc, iddData.nEnergySamples, iddData.nEnergies));
    cudaErrchk(cudaMemcpyToArray(devNucSqSigmaArr, 0, 0, &iddData.nucSqSigmaMatrix[0], iddData.nEnergySamples*iddData.nEnergies*sizeof(float), cudaMemcpyHostToDevice));
    #if CUDART_VERSION < 12000
    nucSqSigmaTex.normalized = false;
    nucSqSigmaTex.filterMode = cudaFilterModeLinear;
    nucSqSigmaTex.addressMode[0] = cudaAddressModeClamp;
    nucSqSigmaTex.addressMode[1] = cudaAddressModeClamp;
    cudaErrchk(cudaBindTextureToArray(nucSqSigmaTex, devNucSqSigmaArr, floatChannelDesc));
    #else
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = devNucSqSigmaArr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = iddData.nEnergySamples*iddData.nEnergies*sizeof(float);
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    cudaTextureObject_t nucSqSigmaTex=0;
    cudaCreateTextureObject(&nucSqSigmaTex, &resDesc, &texDesc, NULL);
    #endif
#endif // NUCLEAR_CORR


#ifdef FINE_GRAINED_TIMING
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeCopyAndBind, start, stop);
    timeTotal += timeCopyAndBind;
    outStream << "    Copy data to GPU and bind to textures: " << timeCopyAndBind << " ms\n\n";
#endif // FINE_GRAINED_TIMING

    for (unsigned int beamNo=0; beamNo<beams.size(); ++beamNo)
    {

#ifdef FINE_GRAINED_TIMING
        outStream << "    Calculating field no. " << beamNo << "\n";
        timeFillIddSigma = 0.0f;
        timePrepareSuperp = 0.0f;
        timeSuperp = 0.0f;
        cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING

        BeamSettings beam = beams[beamNo];
        const size_t nLayers = beam.getEnergies().size();
        std::vector<float> xSigmas(nLayers);
        std::vector<float> ySigmas(nLayers);
        for (unsigned int layerNo=0; layerNo<nLayers; ++layerNo) {
            xSigmas[layerNo] = beam.getSpotSigmas()[layerNo].x;
            ySigmas[layerNo] = beam.getSpotSigmas()[layerNo].y;
        }
        const float2 maxSpotSigmas = make_float2(findMax(xSigmas), findMax(ySigmas));
        //const float3 primRayRes = make_float3(beam.getSpotIdxToGantry().getDelta().x/beam.getRaySampling().x,
        //  beam.getSpotIdxToGantry().getDelta().y/beam.getRaySampling().y, beam.getSpotIdxToGantry().getDelta().z); // Spacing between rays at origin in mm
        const float3 primRayRes = make_float3(beam.getRaySpacing().x, beam.getRaySpacing().y, beam.getSpotIdxToGantry().getDelta().z); // Spacing between rays at origin in mm

        /*// Testing, remove
        float maxZ = -1.0f * std::numeric_limits<float>::infinity();
        float minZ = std::numeric_limits<float>::infinity();
        for (int k=0; k<2; ++k) {
            float z = float(k*imVol->getDims().z) - HALF;
            for (int j=0; j<2; ++j) {
                float y = float(j*imVol->getDims().y) - HALF;
                for (int i=0; i<2; ++i) {
                    float x = float(i*imVol->getDims().x) - HALF;
                    float3 cnr = beam.getGantryToImIdx().inverse().transformPoint(make_float3(x,y,z));
                    //outStream << cnr.x << " " << cnr.y << " " << cnr.z << '\n';
                    if ( cnr.z > maxZ) { maxZ = cnr.z; }
                    if ( cnr.z < minZ) { minZ = cnr.z; }
                }
            }
        }
        float steps = ( maxZ - minZ ) / -beam.getSpotIdxToGantry().getDelta().z;
        outStream << "MaxZ: " << maxZ << ", minZ: " << minZ << ", steps: " << steps << "\n\n";*/

        const uint3 spotGridDims = make_uint3(beam.getWeights()->getDims().x, beam.getWeights()->getDims().y, beam.getWeights()->getDims().z);
        const unsigned int spotGridN = spotGridDims.x * spotGridDims.y * spotGridDims.z;

        // We want a coordinate system which includes all rays within the estimated (since the real value of maxSpotSigma depends on entry depth,
        // which is not known yet) maximum convolution radius and is guaranteed to have ray a centred at gantry (0,0).
        // Find distances, in primRayRes steps, between gantry (0,0) and extreme position rays
        int lSteps = int( ceil( ( beam.getSpotIdxToGantry().getOffset().x - (CONV_SIGMA_CUTOFF*maxSpotSigmas.x + HALF*primRayRes.x) ) / primRayRes.x ) );
        int bSteps = int( ceil( ( beam.getSpotIdxToGantry().getOffset().y - (CONV_SIGMA_CUTOFF*maxSpotSigmas.y + HALF*primRayRes.y) ) / primRayRes.y ) );
        int rSteps = int( floor ( ( (spotGridDims.x-1)*beam.getSpotIdxToGantry().getDelta().x + beam.getSpotIdxToGantry().getOffset().x + (CONV_SIGMA_CUTOFF*maxSpotSigmas.x + HALF*primRayRes.x) ) / primRayRes.x ) );
        int tSteps = int( floor ( ( (spotGridDims.y-1)*beam.getSpotIdxToGantry().getDelta().y + beam.getSpotIdxToGantry().getOffset().y + (CONV_SIGMA_CUTOFF*maxSpotSigmas.y + HALF*primRayRes.y) ) / primRayRes.y ) );
        float3 primRayOffset = make_float3( primRayRes.x*lSteps, primRayRes.y*bSteps, beam.getSpotIdxToGantry().getOffset().z );

        Float3IdxTransform primRayIdxToGantry(primRayRes, primRayOffset);
        Float3FromFanTransform rayIdxToImIdx(primRayIdxToGantry, beam.getSourceDist(), beam.getGantryToImIdx());

        const uint3 primRayDims = make_uint3( roundTo(rSteps-lSteps+1, superpTileX), roundTo(tSteps-bSteps+1, superpTileY), beam.getWeights()->getDims().z);
        const unsigned int convIntermN = primRayDims.x * spotGridDims.y * spotGridDims.z;
        const unsigned int rayWeightsN = primRayDims.x * primRayDims.y * spotGridDims.z;

        const unsigned int rayIddN = primRayDims.x*primRayDims.y*beam.getSteps();

        #ifdef NUCLEAR_CORR
        // Extend nuclear PB map to make divisible by superpTileX and superpTileY, so we can use KS KF
        int3 nucRayDims = make_int3(roundTo(spotGridDims.x, superpTileX), roundTo(spotGridDims.y, superpTileY), spotGridDims.z);
        int nucIddN = nucRayDims.x * nucRayDims.y * beam.getSteps();
        #endif

        //const int3 iddDim = make_int3(beam.getWeights()->getDims().x, beam.getWeights()->getDims().y, beam.getSteps());
        dim3 tracerBlock(32, 8); // Both desktop and laptop
        dim3 tracerGrid(primRayDims.x/tracerBlock.x, primRayDims.y/tracerBlock.y);
        const unsigned int tracerThreadN = primRayDims.x*primRayDims.y;

        const unsigned int weplMinBlockSize = 128;
        dim3 weplMinGridDim(1, 1, beam.getSteps());

        dim3 superpBlockDim(superpTileX, 8); // Desktop and laptop
        //dim3 superpScatGridDim((primRayDims.x+superpTileX-1)/superpTileX, (primRayDims.y+superpTileY-1)/superpTileY, 1); // Todo: primRayDims.x will alway be divisible by superpTileX, right? Corresponding for primRayDims.y?

        const int tileRadBlockY = 4;
        dim3 tileRadBlockDim(superpTileX, tileRadBlockY);

        float *devSpotWeights; // Spot weights
        cudaErrchk(cudaMalloc((void**)&devSpotWeights, spotGridN*sizeof(float)));

        float *devConvInterm; // Intermediate when performing 2D convolution of spot weights
        cudaErrchk(cudaMalloc((void**)&devConvInterm, convIntermN*sizeof(float)));

        float *devPrimRayWeights; // Ray weights
        cudaErrchk(cudaMalloc((void**)&devPrimRayWeights, rayWeightsN*sizeof(float)));

        float2 *devEntrySigmas; // Sigmas for the different energies at global entry depth
        cudaErrchk(cudaMalloc((void**)&devEntrySigmas, spotGridDims.z*sizeof(float2)));

        float *devPrimIdd; // Ray dose before kernel superposition
        cudaErrchk(cudaMalloc((void**)&devPrimIdd, rayIddN*sizeof(float)));

        float *devPrimRSigmaEff; // Reciprocal of effective sigmas
        cudaErrchk(cudaMalloc((void**)&devPrimRSigmaEff, rayIddN*sizeof(float)));

        float *devBevDensity; // Mass density at voxel centre
        cudaErrchk(cudaMalloc((void**)&devBevDensity, rayIddN*sizeof(float)));

        float *devBevWepl; // WEPL to far end of voxel
        cudaErrchk(cudaMalloc((void**)&devBevWepl, rayIddN*sizeof(float)));

        int *devRayFirstInside; // Step number (compared to the global entry depth) where each ray first enters the patient
        cudaErrchk(cudaMalloc((void**)&devRayFirstInside, tracerThreadN*sizeof(int)));

        int *devRayFirstOutside; // Step numnber (compared to the global entry depth) where each ray exits the patient
        cudaErrchk(cudaMalloc((void**)&devRayFirstOutside, tracerThreadN*sizeof(int)));

        int *devRayFirstPassive; // Step number (compared to the global entry depth) where each ray is no longer live
        cudaErrchk(cudaMalloc((void**)&devRayFirstPassive, tracerThreadN*sizeof(int)));

        int *devBeamFirstInside; // Step number (compared to the global entry depth) where a ray of the current beam first enters the patient
        cudaErrchk(cudaMalloc((void**)&devBeamFirstInside, sizeof(int)));

        int *devBeamFirstOutside; // Step number (compared to the global entry depth) where a ray of the current beam last exits the patient
        cudaErrchk(cudaMalloc((void**)&devBeamFirstOutside, sizeof(int)));

        int *devLayerFirstPassive; // Step number (compared to the global entry depth) at which no rays of the current layer are live
        cudaErrchk(cudaMalloc((void**)&devLayerFirstPassive, sizeof(int)));

        float *devWeplMin; // Smallest WEPL in each layer of the BEV
        cudaErrchk(cudaMalloc((void**)&devWeplMin, beam.getSteps()*sizeof(float)));

        float *devRSigmaEffMin; // Smallest reciprocal of sigma (largest sigma) in each layer of the BEV
        cudaErrchk(cudaMalloc((void**)&devRSigmaEffMin, beam.getSteps()*sizeof(float)));

        int *devTilePrimRadCtrs; // tilePrimRadCtrs[rad] holds the number of tiles with max radius rad
        cudaErrchk(cudaMalloc((void**)&devTilePrimRadCtrs, (maxSuperpR+2)*sizeof(int)));

#ifdef NUCLEAR_CORR
        float *devNucRayWeights; // Weights 'nuclear' rays
        cudaErrchk(cudaMalloc((void**)&devNucRayWeights, nucIddN*sizeof(float)));

        float *devNucIdd; // Dose 'nuclear' rays before kernel superposition
        cudaErrchk(cudaMalloc((void**)&devNucIdd, nucIddN*sizeof(float)));

        float *devNucRSigmaEff; // Reciprocal of effective nuclear sigmas
        cudaErrchk(cudaMalloc((void**)&devNucRSigmaEff, nucIddN*sizeof(float)));

        int *devNucSpotIdx; // Map of nuclear PB indices corresponding to the indices of computational PBs
        cudaErrchk(cudaMalloc((void**)&devNucSpotIdx, primRayDims.x*primRayDims.y*sizeof(int)));

        int *devTileNucRadCtrs; // tilePrimRadCtrs[rad] holds the number of tiles with max radius rad
        cudaErrchk(cudaMalloc((void**)&devTileNucRadCtrs, (maxSuperpR+2)*sizeof(int)));
#endif // NUCLEAR_CORR

#ifdef FINE_GRAINED_TIMING
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeAllocateAndSetup, start, stop);
        float3 testO = primRayIdxToGantry.getOffset();
        float3 testD = primRayIdxToGantry.getDelta();
        outStream << "        Calculated primRayIdxToGantry:\n";
        outStream << "            " << testO.x << " " << testO.y << " " << testO.z << '\n';
        outStream << "            " << testD.x << " " << testD.y << " " << testD.z << '\n';

        cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING

        DensityAndSpTracerParams tracerParams(iddData.densityScaleFact, iddData.spScaleFact, beam.getSteps(), rayIdxToImIdx);
        fillBevDensityAndSp<<<tracerGrid,tracerBlock>>>(devBevDensity, devBevWepl, devRayFirstInside, devRayFirstOutside, tracerParams
        #if CUDART_VERSION >= 12000
        , imVolTex, densityTex, stoppingPowerTex
        #endif
        );

#ifdef FINE_GRAINED_TIMING
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeRaytracing, start, stop);

        cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING

        sliceMinVar<int, 1024> <<<1, 1024, 1024*sizeof(int)>>> (devRayFirstInside, devBeamFirstInside, tracerThreadN);
        int beamFirstInside; // k-index of first non-air voxel
        cudaErrchk(cudaMemcpy(&beamFirstInside, devBeamFirstInside, sizeof(int), cudaMemcpyDeviceToHost));
        float entryZ = (float (beamFirstInside)) * primRayRes.z + primRayOffset.z;
        sliceMaxVar<int, 1024> <<<1, 1024, 1024*sizeof(int)>>> (devRayFirstOutside, devBeamFirstOutside, tracerThreadN);
        int beamFirstOutside; // k-index of voxel after last non-air voxel
        cudaErrchk(cudaMemcpy(&beamFirstOutside, devBeamFirstOutside, sizeof(int), cudaMemcpyDeviceToHost));
        sliceMinVar<float, weplMinBlockSize> <<<weplMinGridDim, weplMinBlockSize, weplMinBlockSize*sizeof(float)>>> (devBevWepl, devWeplMin, tracerThreadN);
        std::vector<float> weplMin(beam.getSteps());
        cudaErrchk(cudaMemcpy(&weplMin[0], devWeplMin, beam.getSteps()*sizeof(float), cudaMemcpyDeviceToHost));

        float maxEnergy = findMax<float> (beam.getEnergies());
        float maxEnergyIdx = findDecimalOrdered<float, float> (iddData.energiesPerU, maxEnergy);
        float maxPeakDepth = vectorInterpolate<float,float> (iddData.peakDepths, maxEnergyIdx);
        int firstPastCutoffAll = findFirstLargerOrdered<float>(weplMin, BP_DEPTH_CUTOFF*maxPeakDepth);
        const int beamFirstGuaranteedPassive = min(firstPastCutoffAll, beamFirstOutside);
        int beamFirstCalculatedPassive = 0;
        const int maxNoStepsInside = beamFirstGuaranteedPassive - beamFirstInside;
        const int maxNoPrimTiles = maxNoStepsInside * primRayDims.x/superpTileX * primRayDims.y/superpTileY;

        const uint3 bevPrimDoseDim = make_uint3(primRayDims.x+2*maxSuperpR, primRayDims.y+2*maxSuperpR, beamFirstGuaranteedPassive); // Todo: change z dim to globaFirstAllPassive-beamFirstInside?
        unsigned int bevPrimDoseN = bevPrimDoseDim.x*bevPrimDoseDim.y*bevPrimDoseDim.z;

        float *devBevPrimDose;
        cudaErrchk(cudaMalloc((void**)&devBevPrimDose, bevPrimDoseN*sizeof(float)));

        int2 *devPrimInOutIdcs;
        cudaErrchk(cudaMalloc((void**)&devPrimInOutIdcs, (maxSuperpR+2)*maxNoPrimTiles*sizeof(int2)));

        //cudaArray *devBevPrimDoseArr;
        //const cudaExtent bevPrimDoseExt = make_cudaExtent(bevPrimDoseDim.x, bevPrimDoseDim.y, bevPrimDoseDim.z);
        //cudaErrchk(cudaMalloc3DArray(&devBevPrimDoseArr, &floatChannelDesc, bevPrimDoseExt));
        //cudaMemcpy3DParms primDoseCopyParams = {0};
        //primDoseCopyParams.srcPtr = make_cudaPitchedPtr((void*)devBevPrimDose, bevPrimDoseExt.width*sizeof(float), bevPrimDoseExt.width, bevPrimDoseExt.height);
        //primDoseCopyParams.dstArray = devBevPrimDoseArr;
        //primDoseCopyParams.extent = bevPrimDoseExt;
        //primDoseCopyParams.kind = cudaMemcpyDeviceToDevice;
        //bevPrimDoseTex.normalized = false;
        //bevPrimDoseTex.filterMode = cudaFilterModeLinear;
        //bevPrimDoseTex.addressMode[0] = cudaAddressModeBorder;
        //bevPrimDoseTex.addressMode[1] = cudaAddressModeBorder;
        //bevPrimDoseTex.addressMode[2] = cudaAddressModeBorder;

        dim3 fillBlockDim(32, 8); // Desktop? and laptop?
        dim3 fillGridDim((bevPrimDoseDim.x + fillBlockDim.x - 1)/fillBlockDim.x, (bevPrimDoseDim.y + fillBlockDim.y - 1)/fillBlockDim.y);

        fillDevMem<float> <<<fillGridDim, fillBlockDim>>> (devBevPrimDose, bevPrimDoseN, 0.0f);

        std::vector<float> energyIdcs(nLayers);
        std::vector<float> energyScaleFacts(nLayers);
        std::vector<float> peakDepths(nLayers);
        std::vector<float2> entrySigmas(nLayers);
        for (unsigned int layerNo=0; layerNo<nLayers; ++layerNo) {
            float energyPerU = beam.getEnergies()[layerNo];
            energyIdcs[layerNo] = findDecimalOrdered<float, float> (iddData.energiesPerU, energyPerU);
            energyScaleFacts[layerNo] = vectorInterpolate<float,float> (iddData.scaleFacts, energyIdcs[layerNo]);
            peakDepths[layerNo] = vectorInterpolate<float,float> (iddData.peakDepths, energyIdcs[layerNo]);
            float2 sigmaSqCoefs = FillIddAndSigmaParams::sigmaSqAirCoefs(peakDepths[layerNo]);
            float2 spotSigma = beam.getSpotSigmas()[layerNo]; // Previously used empirical fit: 2.3f + 290.0f/(peakDepth+15.0f);
            entrySigmas[layerNo] =  make_float2( sqrtf(sigmaSqCoefs.x*entryZ*entryZ + sigmaSqCoefs.y*entryZ + spotSigma.x*spotSigma.x),
                sqrtf(sigmaSqCoefs.x*entryZ*entryZ + sigmaSqCoefs.y*entryZ + spotSigma.y*spotSigma.y) );
#ifdef NUCLEAR_CORR
#if NUCLEAR_CORR == GAUSS_FIT
            entrySigmas[layerNo].x = 0.97f * entrySigmas[layerNo].x;
            entrySigmas[layerNo].y = 0.97f * entrySigmas[layerNo].y;
#endif
#endif // NUCLEAR_CORR
        }
        float2 pxSpMult = make_float2(1.0f - entryZ/beam.getSourceDist().x, 1.0f - entryZ/beam.getSourceDist().y);

        cudaErrchk(cudaMemcpy(devSpotWeights, beam.getWeights()->getImData(), spotGridN*sizeof(float), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(devEntrySigmas, &entrySigmas[0], spotGridDims.z*sizeof(float2), cudaMemcpyHostToDevice));
        gpuConvolution2D(devSpotWeights, devConvInterm, devPrimRayWeights, devEntrySigmas, spotGridDims, primRayDims, beam.getSpotIdxToGantry().getDelta(),
            beam.getSpotIdxToGantry().getOffset(), primRayRes, primRayOffset, pxSpMult);
        //Delete weights, we do not need any more
        delete beam.getWeights();

#ifdef NUCLEAR_CORR

        // Must fill with zeros since, after extending the nuclear PB map to be divisible by superpTileX and superpTileY,
        // not all nuclear PBs necesarily map to a computational PB
        fillDevMem<float> <<<nucIddN/256, 256>>> (devNucIdd, nucIddN, 0.0f);
        fillDevMem<float> <<<nucIddN/256, 256>>> (devNucRSigmaEff, nucIddN, std::numeric_limits<float>::infinity());

        dim3 extendBlockDims(superpTileX, superpTileY);
        dim3 extendGridDims(nucRayDims.x/superpTileX, nucRayDims.y/superpTileY, nucRayDims.z);
        extendAndPadd <<<extendGridDims, extendBlockDims>>> (devSpotWeights, devNucRayWeights, spotGridDims);

        const int maxNoNucTiles = maxNoStepsInside * nucRayDims.x/superpTileX * nucRayDims.y/superpTileY;

        int2 *devNucInOutIdcs;
        cudaErrchk(cudaMalloc((void**)&devNucInOutIdcs, (maxSuperpR+2)*maxNoNucTiles*sizeof(int2)));

        const uint3 bevNucDoseDim = make_uint3(nucRayDims.x+2*maxSuperpR, nucRayDims.y+2*maxSuperpR, beamFirstGuaranteedPassive);
        unsigned int bevNucDoseN = bevNucDoseDim.x * bevNucDoseDim.y * bevNucDoseDim.z;

        float *devBevNucDose;
        cudaErrchk(cudaMalloc((void**)&devBevNucDose, bevNucDoseN*sizeof(float)));

        fillDevMem<float> <<<bevNucDoseN/256, 256>>> (devBevNucDose, bevNucDoseN, 0.0f);

        std::vector<int> nucSpotIdx( primRayDims.x*primRayDims.y, -1);
        for (int spotIdxY=0; spotIdxY<spotGridDims.y; ++spotIdxY) {
            float gantryPosY = spotIdxY * beam.getSpotIdxToGantry().getDelta().y + beam.getSpotIdxToGantry().getOffset().y;
            int rayIdxY = (int) round( ( gantryPosY - primRayIdxToGantry.getOffset().y ) / primRayIdxToGantry.getDelta().y );
            for (int spotIdxX=0; spotIdxX<spotGridDims.x; ++spotIdxX) {
                float gantryPosX = spotIdxX * beam.getSpotIdxToGantry().getDelta().x + beam.getSpotIdxToGantry().getOffset().x;
                int rayIdxX = (int) round( ( gantryPosX - primRayIdxToGantry.getOffset().x ) / primRayIdxToGantry.getDelta().x );
                nucSpotIdx[primRayDims.x * rayIdxY + rayIdxX] = nucRayDims.x * spotIdxY + spotIdxX;
            }
        }
        cudaErrchk(cudaMemcpy(devNucSpotIdx, &nucSpotIdx[0], primRayDims.x*primRayDims.y*sizeof(int), cudaMemcpyHostToDevice));

        //cudaArray *devBevNucDoseArr;
        //const cudaExtent bevNucDoseExt = make_cudaExtent(bevNucDoseDim.x, bevNucDoseDim.y, bevNucDoseDim.z);
        //cudaErrchk(cudaMalloc3DArray(&devBevNucDoseArr, &floatChannelDesc, bevNucDoseExt));
        //cudaMemcpy3DParms nucDoseCopyParams = {0};
        //nucDoseCopyParams.srcPtr = make_cudaPitchedPtr((void*)devBevNucDose, bevNucDoseExt.width*sizeof(float), bevNucDoseExt.width, bevNucDoseExt.height);
        //nucDoseCopyParams.dstArray = devBevNucDoseArr;
        //nucDoseCopyParams.extent = bevNucDoseExt;
        //nucDoseCopyParams.kind = cudaMemcpyDeviceToDevice;
        //bevNucDoseTex.normalized = false;
        //bevNucDoseTex.filterMode = cudaFilterModeLinear;
        //bevNucDoseTex.addressMode[0] = cudaAddressModeBorder;
        //bevNucDoseTex.addressMode[1] = cudaAddressModeBorder;
        //bevNucDoseTex.addressMode[2] = cudaAddressModeBorder;
#endif // NUCLEAR_CORR

#ifdef FINE_GRAINED_TIMING
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timePrepareEnergyLoop, start, stop);
#endif // FINE_GRAINED_TIMING


        for (unsigned int layerNo = 0; layerNo<nLayers; ++layerNo)
        {

#ifdef FINE_GRAINED_TIMING
            cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING
            float spotDistInRays = beam.getSpotIdxToGantry().getDelta().x / beam.getRaySpacing().x;
            unsigned int localAfterLastStep = findFirstLargerOrdered<float> (weplMin, BP_DEPTH_CUTOFF*peakDepths[layerNo]);
            unsigned int afterLastStep = min(localAfterLastStep, beamFirstGuaranteedPassive);
            FillIddAndSigmaParams fillParams(energyIdcs[layerNo], energyScaleFacts[layerNo], peakDepths[layerNo], entrySigmas[layerNo].x*entrySigmas[layerNo].x, iddData.rRlScaleFact, spotDistInRays, 0, beamFirstInside, afterLastStep, rayIdxToImIdx);
#ifdef NUCLEAR_CORR
            fillIddAndSigma <<<tracerGrid, tracerBlock >>>(devBevDensity, devBevWepl, devPrimIdd, devPrimRSigmaEff, devPrimRayWeights + layerNo*primRayDims.x*primRayDims.y, devNucIdd, devNucRSigmaEff, devNucRayWeights + layerNo*nucRayDims.x*nucRayDims.y, devNucSpotIdx, devRayFirstInside, devRayFirstOutside, devRayFirstPassive, fillParams
            #if CUDART_VERSION >= 12000
            , cumulIddTex, rRadiationLengthTex, nucWeightTex, nucSqSigmaTex
            #endif
            );
#else // NUCLEAR_CORR
            fillIddAndSigma <<<tracerGrid, tracerBlock >>>(devBevDensity, devBevWepl, devPrimIdd, devPrimRSigmaEff, devPrimRayWeights + layerNo*primRayDims.x*primRayDims.y, devRayFirstInside, devRayFirstOutside, devRayFirstPassive, fillParams
            #if CUDART_VERSION >= 12000
            , cumulIddTex, rRadiationLengthTex
            #endif
            );
#endif // NUCLEAR_CORR

#ifdef FINE_GRAINED_TIMING
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            timeFillIddSigma += elapsedTime;

            cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING

            //sliceMinVar<float, weplMinBlockSize> <<<dim3(1,1,afterLastStep), weplMinBlockSize, weplMinBlockSize*sizeof(float)>>> (devPrimRSigmaEff, devRSigmaEffMin, tracerThreadN);
            //std::vector<float> rSigmaEffMin(afterLastStep);
            //cudaErrchk(cudaMemcpy(&rSigmaEffMin[0], devRSigmaEffMin, afterLastStep*sizeof(float), cudaMemcpyDeviceToHost));
            sliceMaxVar<int, 1024> <<<1, 1024, 1024*sizeof(int)>>>(devRayFirstPassive, devLayerFirstPassive, tracerThreadN);
            int layerFirstPassive;
            cudaErrchk(cudaMemcpy(&layerFirstPassive, devLayerFirstPassive, sizeof(int), cudaMemcpyDeviceToHost));
            if (layerFirstPassive > beamFirstCalculatedPassive) {
                beamFirstCalculatedPassive = layerFirstPassive;
            }

            std::vector<int> tilePrimRadCtrs(maxSuperpR+2, 0);
            cudaErrchk(cudaMemcpy(devTilePrimRadCtrs, &tilePrimRadCtrs[0], (maxSuperpR+2)*sizeof(int), cudaMemcpyHostToDevice));
            dim3 tilePrimRadGridDim(primRayDims.x/superpTileX, primRayDims.y/superpTileY, layerFirstPassive-beamFirstInside);
            tileRadCalc<tileRadBlockY> <<<tilePrimRadGridDim, tileRadBlockDim>>> (devPrimRSigmaEff, beamFirstInside, devTilePrimRadCtrs, devPrimInOutIdcs, maxNoPrimTiles);
            cudaErrchk(cudaMemcpy(&tilePrimRadCtrs[0], devTilePrimRadCtrs, (maxSuperpR+2)*sizeof(int), cudaMemcpyDeviceToHost));

            if (tilePrimRadCtrs[maxSuperpR+1] > 0) { throw("Found larger than allowed kernel superposition radius"); }
            int layerMaxPrimSuperpR = 0;
            for (unsigned int i=0; i<maxSuperpR+2; ++i) { if( tilePrimRadCtrs[i]>0 ) { layerMaxPrimSuperpR=i; }  }
            int recPrimRad = layerMaxPrimSuperpR;
            std::vector<int> batchedPrimTileRadCtrs(maxSuperpR+1, 0);
            batchedPrimTileRadCtrs[0] = tilePrimRadCtrs[0];
            for (int rad=layerMaxPrimSuperpR; rad>0; --rad) {
                batchedPrimTileRadCtrs[recPrimRad] += tilePrimRadCtrs[rad];
                if (batchedPrimTileRadCtrs[recPrimRad] >= minTilesInBatch) {
                    recPrimRad = rad-1;
                }
            }

#ifdef NUCLEAR_CORR
            std::vector<int> tileNucRadCtrs(maxSuperpR+2, 0);
            cudaErrchk(cudaMemcpy(devTileNucRadCtrs, &tileNucRadCtrs[0], (maxSuperpR+2)*sizeof(int), cudaMemcpyHostToDevice));
            dim3 tileNucRadGridDim(nucRayDims.x/superpTileX, nucRayDims.y/superpTileY, layerFirstPassive-beamFirstInside);
            tileRadCalc<tileRadBlockY> <<<tileNucRadGridDim, tileRadBlockDim>>> (devNucRSigmaEff, beamFirstInside, devTileNucRadCtrs, devNucInOutIdcs, maxNoNucTiles);
            cudaErrchk(cudaMemcpy(&tileNucRadCtrs[0], devTileNucRadCtrs, (maxSuperpR+2)*sizeof(int), cudaMemcpyDeviceToHost));

            if (tileNucRadCtrs[maxSuperpR+1] > 0) { throw("Found larger than allowed kernel superposition radius"); }
            int layerMaxNucSuperpR = 0;
            for (int i=0; i<maxSuperpR+2; ++i) { if( tileNucRadCtrs[i]>0 ) { layerMaxNucSuperpR=i; }  }
            int recNucRad = layerMaxNucSuperpR;
            std::vector<int> batchedNucTileRadCtrs(maxSuperpR+1, 0);
            batchedNucTileRadCtrs[0] = tileNucRadCtrs[0];
            for (int rad=layerMaxNucSuperpR; rad>0; --rad) {
                batchedNucTileRadCtrs[recNucRad] += tileNucRadCtrs[rad];
                if (batchedNucTileRadCtrs[recNucRad] >= minTilesInBatch) {
                    recNucRad = rad-1;
                }
            }
#endif // NUCLEAR_CORR

#ifdef FINE_GRAINED_TIMING
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            timePrepareSuperp += elapsedTime;

            //for (int i=0; i<maxSuperpR+2; ++i) { outStream << tilePrimRadCtrs[i] << ' '; }
            //outStream << '\n';
            //for (int i=0; i<maxSuperpR+1; ++i) { outStream << batchedPrimTileRadCtrs[i] << ' '; }
            //outStream << '\n';

#ifdef NUCLEAR_CORR
            //for (int i=0; i<maxSuperpR+2; ++i) { outStream << tileNucRadCtrs[i] << ' '; }
            //outStream << '\n';
            //for (int i=0; i<maxSuperpR+1; ++i) { outStream << batchedNucTileRadCtrs[i] << ' '; }
            //outStream << '\n';
#endif // NUCLEAR_CORR

            outStream << "        Layer: " << layerNo << ", energy idx: " << energyIdcs[layerNo] << ", peak depth: " << peakDepths[layerNo] << ", steps: " << layerFirstPassive-beamFirstInside
                << "\n            entry step: " << beamFirstInside << ", entry sigmas: (" << entrySigmas[layerNo].x << ", " <<  entrySigmas[layerNo].y
                << "), max radius: " << layerMaxPrimSuperpR << '\n';

            cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING

            if (batchedPrimTileRadCtrs[0] > 0) { kernelSuperposition<0> <<<batchedPrimTileRadCtrs[0], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[1] > 0) { kernelSuperposition<1> <<<batchedPrimTileRadCtrs[1], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[2] > 0) { kernelSuperposition<2> <<<batchedPrimTileRadCtrs[2], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[3] > 0) { kernelSuperposition<3> <<<batchedPrimTileRadCtrs[3], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[4] > 0) { kernelSuperposition<4> <<<batchedPrimTileRadCtrs[4], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[5] > 0) { kernelSuperposition<5> <<<batchedPrimTileRadCtrs[5], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[6] > 0) { kernelSuperposition<6> <<<batchedPrimTileRadCtrs[6], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[7] > 0) { kernelSuperposition<7> <<<batchedPrimTileRadCtrs[7], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[8] > 0) { kernelSuperposition<8> <<<batchedPrimTileRadCtrs[8], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[9] > 0) { kernelSuperposition<9> <<<batchedPrimTileRadCtrs[9], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[10] > 0) { kernelSuperposition<10> <<<batchedPrimTileRadCtrs[10], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[11] > 0) { kernelSuperposition<11> <<<batchedPrimTileRadCtrs[11], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[12] > 0) { kernelSuperposition<12> <<<batchedPrimTileRadCtrs[12], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[13] > 0) { kernelSuperposition<13> <<<batchedPrimTileRadCtrs[13], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[14] > 0) { kernelSuperposition<14> <<<batchedPrimTileRadCtrs[14], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[15] > 0) { kernelSuperposition<15> <<<batchedPrimTileRadCtrs[15], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[16] > 0) { kernelSuperposition<16> <<<batchedPrimTileRadCtrs[16], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[17] > 0) { kernelSuperposition<17> <<<batchedPrimTileRadCtrs[17], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[18] > 0) { kernelSuperposition<18> <<<batchedPrimTileRadCtrs[18], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[19] > 0) { kernelSuperposition<19> <<<batchedPrimTileRadCtrs[19], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[20] > 0) { kernelSuperposition<20> <<<batchedPrimTileRadCtrs[20], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[21] > 0) { kernelSuperposition<21> <<<batchedPrimTileRadCtrs[21], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[22] > 0) { kernelSuperposition<22> <<<batchedPrimTileRadCtrs[22], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[23] > 0) { kernelSuperposition<23> <<<batchedPrimTileRadCtrs[23], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[24] > 0) { kernelSuperposition<24> <<<batchedPrimTileRadCtrs[24], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[25] > 0) { kernelSuperposition<25> <<<batchedPrimTileRadCtrs[25], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[26] > 0) { kernelSuperposition<26> <<<batchedPrimTileRadCtrs[26], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[27] > 0) { kernelSuperposition<27> <<<batchedPrimTileRadCtrs[27], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[28] > 0) { kernelSuperposition<28> <<<batchedPrimTileRadCtrs[28], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[29] > 0) { kernelSuperposition<29> <<<batchedPrimTileRadCtrs[29], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[30] > 0) { kernelSuperposition<30> <<<batchedPrimTileRadCtrs[30], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[31] > 0) { kernelSuperposition<31> <<<batchedPrimTileRadCtrs[31], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }
            if (batchedPrimTileRadCtrs[32] > 0) { kernelSuperposition<32> <<<batchedPrimTileRadCtrs[32], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); }

#ifdef NUCLEAR_CORR
            if (batchedNucTileRadCtrs[0] > 0) { kernelSuperposition<0> <<<batchedNucTileRadCtrs[0], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[1] > 0) { kernelSuperposition<1> <<<batchedNucTileRadCtrs[1], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[2] > 0) { kernelSuperposition<2> <<<batchedNucTileRadCtrs[2], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[3] > 0) { kernelSuperposition<3> <<<batchedNucTileRadCtrs[3], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[4] > 0) { kernelSuperposition<4> <<<batchedNucTileRadCtrs[4], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[5] > 0) { kernelSuperposition<5> <<<batchedNucTileRadCtrs[5], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[6] > 0) { kernelSuperposition<6> <<<batchedNucTileRadCtrs[6], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[7] > 0) { kernelSuperposition<7> <<<batchedNucTileRadCtrs[7], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[8] > 0) { kernelSuperposition<8> <<<batchedNucTileRadCtrs[8], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[9] > 0) { kernelSuperposition<9> <<<batchedNucTileRadCtrs[9], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[10] > 0) { kernelSuperposition<10> <<<batchedNucTileRadCtrs[10], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[11] > 0) { kernelSuperposition<11> <<<batchedNucTileRadCtrs[11], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[12] > 0) { kernelSuperposition<12> <<<batchedNucTileRadCtrs[12], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[13] > 0) { kernelSuperposition<13> <<<batchedNucTileRadCtrs[13], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[14] > 0) { kernelSuperposition<14> <<<batchedNucTileRadCtrs[14], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[15] > 0) { kernelSuperposition<15> <<<batchedNucTileRadCtrs[15], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[16] > 0) { kernelSuperposition<16> <<<batchedNucTileRadCtrs[16], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[17] > 0) { kernelSuperposition<17> <<<batchedNucTileRadCtrs[17], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[18] > 0) { kernelSuperposition<18> <<<batchedNucTileRadCtrs[18], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[19] > 0) { kernelSuperposition<19> <<<batchedNucTileRadCtrs[19], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[20] > 0) { kernelSuperposition<20> <<<batchedNucTileRadCtrs[20], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[21] > 0) { kernelSuperposition<21> <<<batchedNucTileRadCtrs[21], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[22] > 0) { kernelSuperposition<22> <<<batchedNucTileRadCtrs[22], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[23] > 0) { kernelSuperposition<23> <<<batchedNucTileRadCtrs[23], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[24] > 0) { kernelSuperposition<24> <<<batchedNucTileRadCtrs[24], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[25] > 0) { kernelSuperposition<25> <<<batchedNucTileRadCtrs[25], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[26] > 0) { kernelSuperposition<26> <<<batchedNucTileRadCtrs[26], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[27] > 0) { kernelSuperposition<27> <<<batchedNucTileRadCtrs[27], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[28] > 0) { kernelSuperposition<28> <<<batchedNucTileRadCtrs[28], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[29] > 0) { kernelSuperposition<29> <<<batchedNucTileRadCtrs[29], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[30] > 0) { kernelSuperposition<30> <<<batchedNucTileRadCtrs[30], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[31] > 0) { kernelSuperposition<31> <<<batchedNucTileRadCtrs[31], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
            if (batchedNucTileRadCtrs[32] > 0) { kernelSuperposition<32> <<<batchedNucTileRadCtrs[32], superpBlockDim>>>(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, devNucInOutIdcs, maxNoNucTiles, devTileNucRadCtrs); }
#endif // NUCLEAR_CORR

#ifdef FINE_GRAINED_TIMING
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            timeSuperp += elapsedTime;
#endif // FINE_GRAINED_TIMING
        }

#ifdef FINE_GRAINED_TIMING

        cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING

        cudaArray *devBevPrimDoseArr;
        const cudaExtent bevPrimDoseExt = make_cudaExtent(bevPrimDoseDim.x, bevPrimDoseDim.y, beamFirstCalculatedPassive-beamFirstInside);
        cudaErrchk(cudaMalloc3DArray(&devBevPrimDoseArr, &floatChannelDesc, bevPrimDoseExt));
        cudaMemcpy3DParms primDoseCopyParams = {};
        primDoseCopyParams.srcPtr = make_cudaPitchedPtr((void*)(devBevPrimDose + beamFirstInside*bevPrimDoseDim.x*bevPrimDoseDim.y), bevPrimDoseExt.width*sizeof(float), bevPrimDoseExt.width, bevPrimDoseExt.height);
        primDoseCopyParams.dstArray = devBevPrimDoseArr;
        primDoseCopyParams.extent = bevPrimDoseExt;
        primDoseCopyParams.kind = cudaMemcpyDeviceToDevice;
        #if CUDART_VERSION < 12000
        bevPrimDoseTex.normalized = false;
        bevPrimDoseTex.filterMode = cudaFilterModeLinear;
        bevPrimDoseTex.addressMode[0] = cudaAddressModeBorder;
        bevPrimDoseTex.addressMode[1] = cudaAddressModeBorder;
        bevPrimDoseTex.addressMode[2] = cudaAddressModeBorder;
        #else
        // create texture object
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = devBevPrimDoseArr;
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = bevPrimDoseExt.width*bevPrimDoseExt.height*bevPrimDoseExt.depth*sizeof(float);
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.addressMode[2] = cudaAddressModeBorder;
        cudaTextureObject_t bevPrimDoseTex=0;
        cudaCreateTextureObject(&bevPrimDoseTex, &resDesc, &texDesc, NULL);
        #endif
        cudaErrchk(cudaMemcpy3D(&primDoseCopyParams));
        #if CUDART_VERSION < 12000
        cudaErrchk(cudaBindTextureToArray(bevPrimDoseTex, devBevPrimDoseArr, floatChannelDesc));
        #endif

#ifdef NUCLEAR_CORR
        cudaArray *devBevNucDoseArr;
        const cudaExtent bevNucDoseExt = make_cudaExtent(bevNucDoseDim.x, bevNucDoseDim.y, beamFirstCalculatedPassive-beamFirstInside);
        cudaErrchk(cudaMalloc3DArray(&devBevNucDoseArr, &floatChannelDesc, bevNucDoseExt));
        cudaMemcpy3DParms nucDoseCopyParams = {0};
        nucDoseCopyParams.srcPtr = make_cudaPitchedPtr((void*)(devBevNucDose + beamFirstInside*bevNucDoseDim.x*bevNucDoseDim.y), bevNucDoseExt.width*sizeof(float), bevNucDoseExt.width, bevNucDoseExt.height);
        nucDoseCopyParams.dstArray = devBevNucDoseArr;
        nucDoseCopyParams.extent = bevNucDoseExt;
        nucDoseCopyParams.kind = cudaMemcpyDeviceToDevice;
        #if CUDART_VERSION < 12000
        bevNucDoseTex.normalized = false;
        bevNucDoseTex.filterMode = cudaFilterModeLinear;
        bevNucDoseTex.addressMode[0] = cudaAddressModeBorder;
        bevNucDoseTex.addressMode[1] = cudaAddressModeBorder;
        bevNucDoseTex.addressMode[2] = cudaAddressModeBorder;
        #else
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = devBevNucDoseArr;
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = bevNucDoseExt.width*bevNucDoseExt.height*bevNucDoseExt.depth*sizeof(float);
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.addressMode[2] = cudaAddressModeBorder;
        cudaTextureObject_t bevNucDoseTex=0;
        cudaCreateTextureObject(&bevNucDoseTex, &resDesc, &texDesc, NULL);
        #endif
        cudaErrchk(cudaMemcpy3D(&nucDoseCopyParams));
        #if CUDART_VERSION < 12000
        cudaErrchk(cudaBindTextureToArray(bevNucDoseTex, devBevNucDoseArr, floatChannelDesc));
        #endif
#endif // NUCLEAR_CORR

#ifdef FINE_GRAINED_TIMING
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeCopyingToTexture, start, stop);
        cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING

        Float3FromFanTransform primRayIdxToDoseIdx(primRayIdxToGantry, beam.getSourceDist(), beam.getGantryToDoseIdx());

        float3 primTransfPoint;
        float3 primMaxPoint = make_float3(-1.0f, -1.0f, -1.0f);
        float3 primMinPoint = make_float3(100000.0f, 100000.0f, 100000.0f);
        float xVals[2] = { -float(maxSuperpR), float(primRayDims.x + maxSuperpR - 1) };
        float yVals[2] = { -float(maxSuperpR), float(primRayDims.y + maxSuperpR - 1) };
        float zVals[2] = { float(beamFirstInside), float(beamFirstCalculatedPassive - 1) };

        for (int zVal = 0; zVal < 2; ++zVal) {
            for (int yVal = 0; yVal < 2; ++yVal) {
                for (int xVal = 0; xVal < 2; ++xVal) {
                    primTransfPoint = primRayIdxToDoseIdx.transformPoint(make_float3(xVals[xVal], yVals[yVal], zVals[zVal]));
                    if (primTransfPoint.x > primMaxPoint.x) { primMaxPoint.x = primTransfPoint.x; }
                    if (primTransfPoint.y > primMaxPoint.y) { primMaxPoint.y = primTransfPoint.y; }
                    if (primTransfPoint.z > primMaxPoint.z) { primMaxPoint.z = primTransfPoint.z; }
                    if (primTransfPoint.x < primMinPoint.x) { primMinPoint.x = primTransfPoint.x; }
                    if (primTransfPoint.y < primMinPoint.y) { primMinPoint.y = primTransfPoint.y; }
                    if (primTransfPoint.z < primMinPoint.z) { primMinPoint.z = primTransfPoint.z; }
                }
            }
        }
        int3 minIdx = make_int3( max( ((int(floor(primMinPoint.x)))/32) * 32, 0) , max( int(floor(primMinPoint.y)), 0), max( int(floor(primMinPoint.z)),0) );
        int3 maxIdx = make_int3( min( int(ceil(primMaxPoint.x)), doseVol->getDims().x-1), min( int(ceil(primMaxPoint.y)), doseVol->getDims().y-1), min( int(ceil(primMaxPoint.z)), doseVol->getDims().z-1) );
        dim3 transfBlockDim(32, 8); // Desktop and laptop
        dim3 primTransfGridDim(roundTo(maxIdx.x - minIdx.x + 1, transfBlockDim.x)/transfBlockDim.x, roundTo(maxIdx.y - minIdx.y + 1, transfBlockDim.y)/transfBlockDim.y);

        //TransferParamStructDiv3 primTransfStruct(primRayIdxToDoseIdx.invertAndShift(make_float2(float(maxSuperpR), float(maxSuperpR))));
        TransferParamStructDiv3 primTransfStruct(primRayIdxToDoseIdx.invertAndShift(make_float3(float(maxSuperpR), float(maxSuperpR), -float(beamFirstInside))));
        primTransfDiv<<<primTransfGridDim, transfBlockDim>>>(devDoseBox, primTransfStruct, minIdx, maxIdx.z, doseVol->getDims()
        #if CUDART_VERSION >= 12000
        , bevPrimDoseTex
        #endif
        );

#ifdef NUCLEAR_CORR
        Float3FromFanTransform nucRayIdxToDoseIdx(beam.getSpotIdxToGantry(), beam.getSourceDist(), beam.getGantryToDoseIdx());

        float3 nucTransfPoint;
        float3 nucMaxPoint = make_float3(-1.0f, -1.0f, -1.0f);
        float3 nucMinPoint = make_float3(100000.0f, 100000.0f, 100000.0f);
        xVals[0] = -float(maxSuperpR);
        xVals[1] = float(nucRayDims.x + maxSuperpR - 1);
        yVals[0] = -float(maxSuperpR);
        yVals[1] = float(nucRayDims.y + maxSuperpR - 1);

        for (int zVal = 0; zVal < 2; ++zVal) {
            for (int yVal = 0; yVal < 2; ++yVal) {
                for (int xVal = 0; xVal < 2; ++xVal) {
                    nucTransfPoint = nucRayIdxToDoseIdx.transformPoint(make_float3(xVals[xVal], yVals[yVal], zVals[zVal]));
                    if (nucTransfPoint.x > nucMaxPoint.x) { nucMaxPoint.x = nucTransfPoint.x; }
                    if (nucTransfPoint.y > nucMaxPoint.y) { nucMaxPoint.y = nucTransfPoint.y; }
                    if (nucTransfPoint.z > nucMaxPoint.z) { nucMaxPoint.z = nucTransfPoint.z; }
                    if (nucTransfPoint.x < nucMinPoint.x) { nucMinPoint.x = nucTransfPoint.x; }
                    if (nucTransfPoint.y < nucMinPoint.y) { nucMinPoint.y = nucTransfPoint.y; }
                    if (nucTransfPoint.z < nucMinPoint.z) { nucMinPoint.z = nucTransfPoint.z; }
                }
            }
        }
        minIdx = make_int3( max( ((int(floor(nucMinPoint.x)))/32) * 32, 0) , max( int(floor(nucMinPoint.y)), 0), max( int(floor(nucMinPoint.z)),0) );
        maxIdx = make_int3( min( int(ceil(nucMaxPoint.x)), doseVol->getDims().x-1), min( int(ceil(nucMaxPoint.y)), doseVol->getDims().y-1), min( int(ceil(nucMaxPoint.z)), doseVol->getDims().z-1) );
        dim3 nucTransfGridDim(roundTo(maxIdx.x - minIdx.x + 1, transfBlockDim.x)/transfBlockDim.x, roundTo(maxIdx.y - minIdx.y + 1, transfBlockDim.y)/transfBlockDim.y);

        //TransferParamStructDiv3 nucTransfStruct(nucRayIdxToDoseIdx.invertAndShift(make_float2(float(maxSuperpR), float(maxSuperpR))));
        TransferParamStructDiv3 nucTransfStruct(nucRayIdxToDoseIdx.invertAndShift(make_float3(float(maxSuperpR), float(maxSuperpR), -float(beamFirstInside))));
        nucTransfDiv<<<nucTransfGridDim, transfBlockDim>>>(devDoseBox, nucTransfStruct, minIdx, maxIdx.z, doseVol->getDims()
        #if CUDART_VERSION >= 12000
        , bevNucDoseTex
        #endif
        );
#endif // NUCLEAR_CORR


#ifdef FINE_GRAINED_TIMING
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeTransforming, start, stop);
        cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING

        cudaErrchk(cudaFree(devSpotWeights));
        cudaErrchk(cudaFree(devConvInterm));
        cudaErrchk(cudaFree(devPrimRayWeights));
        cudaErrchk(cudaFree(devPrimIdd));
        cudaErrchk(cudaFree(devPrimRSigmaEff));
        cudaErrchk(cudaFree(devBevDensity));
        cudaErrchk(cudaFree(devBevWepl));
        cudaErrchk(cudaFree(devBevPrimDose));
        cudaErrchk(cudaFree(devRayFirstInside));
        cudaErrchk(cudaFree(devRayFirstOutside));
        cudaErrchk(cudaFree(devRayFirstPassive));
        cudaErrchk(cudaFree(devBeamFirstInside));
        cudaErrchk(cudaFree(devBeamFirstOutside));
        cudaErrchk(cudaFree(devLayerFirstPassive));
        cudaErrchk(cudaFree(devWeplMin));
        cudaErrchk(cudaFree(devRSigmaEffMin));
        cudaErrchk(cudaFreeArray(devBevPrimDoseArr));
#ifdef NUCLEAR_CORR
        cudaErrchk(cudaFree(devNucRayWeights));
        cudaErrchk(cudaFree(devNucIdd));
        cudaErrchk(cudaFree(devNucRSigmaEff));
        cudaErrchk(cudaFree(devNucSpotIdx));
        cudaErrchk(cudaFree(devTileNucRadCtrs));
        cudaErrchk(cudaFreeArray(devBevNucDoseArr));
#endif // NUCLEAR_CORR

#ifdef FINE_GRAINED_TIMING
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        timeFreeMem += elapsedTime;

        int transfN = primTransfGridDim.x*primTransfGridDim.y*transfBlockDim.x*transfBlockDim.y*(maxIdx.z-minIdx.z+1);
        outStream << "\n";
        outStream << "        Allocate memory and set up parameters: " << timeAllocateAndSetup << " ms\n";
        outStream << "        Time to trace " << tracerGrid.x*tracerBlock.x << "x" << tracerGrid.y*tracerBlock.y << " rays "
            << beam.getSteps() << " steps: " << timeRaytracing << " ms\n";
        outStream << "        Time preparing data for loop over energies: " << timePrepareEnergyLoop << " ms\n";
        outStream << "        Time depositing IDD and calculating sigma " << nLayers << " time(s): " << timeFillIddSigma << " ms\n";
        outStream << "        Time preparing for superposition " << nLayers << " time(s): "  << timePrepareSuperp << " ms\n";
        outStream << "        Time executing superposition " << nLayers << " time(s): "  << timeSuperp << " ms\n";
        outStream << "        Copy dose distribution to texture memory: " << timeCopyingToTexture << " ms\n";
        outStream << "        Kernel time to transform " << transfN << " voxels: " << timeTransforming << " ms\n\n";

        timeTotal += timeAllocateAndSetup + timeRaytracing + timePrepareEnergyLoop + timeFillIddSigma + timePrepareSuperp
            + timeSuperp + timeCopyingToTexture + timeTransforming;
#endif //FINE_GRAINED_TIMING
    }

#ifdef FINE_GRAINED_TIMING
    cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING

    cudaErrchk(cudaMemcpy(doseVol->getImData(), devDoseBox, doseN*sizeof(float), cudaMemcpyDeviceToHost));

#ifdef FINE_GRAINED_TIMING
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeCopyBack, start, stop);
    timeTotal += timeCopyBack;
    outStream << "    Time to copy dose back to host: " << timeCopyBack << " ms\n";

    //float totalTime = timeCopyAndBind + timeAllocateAndSetup + timeRaytracing + timePrepareEnergyLoop + timeFillIddSigma
    //  + timePrepareSuperp + timeSuperp + timeCopyingToTexture + timeTransforming + timeCopyBack;

    cudaEventRecord(start, 0);
#endif // FINE_GRAINED_TIMING

    cudaErrchk(cudaFree(devDoseBox));
    cudaErrchk(cudaFreeArray(devImVolArr));
    cudaErrchk(cudaFreeArray(devCumulIddArr));
    cudaErrchk(cudaFreeArray(devDensityArr));
    cudaErrchk(cudaFreeArray(devStoppingPowerArr));
    cudaErrchk(cudaFreeArray(devReciprocalRadiationLengthArr));
#ifdef NUCLEAR_CORR
    cudaErrchk(cudaFreeArray(devNucWeightArr));
    cudaErrchk(cudaFreeArray(devNucSqSigmaArr));
#endif // NUCLEAR_CORR

#ifdef FINE_GRAINED_TIMING
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    timeFreeMem += elapsedTime;
    outStream << "    Time spent freeing memory: " << timeFreeMem << " ms.\n\n";
    timeTotal += timeFreeMem;
    outStream << "    Approximate total execution time (excluding GPU initialisation): " << timeTotal << " ms.\n";
    outStream << "    (Remove FINE_GRAINED_TIMING flag for more accurate total time, reports up to 30 ms longer execution time)\n\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

#else // FINE_GRAINED_TIMING
    cudaEventRecord(globalStop, 0);
    cudaEventSynchronize(globalStop);
    cudaEventElapsedTime(&globalTime, globalStart, globalStop);
    outStream << "    Total global execution time (excluding GPU initialisation): " << globalTime << " ms.\n\n";
    cudaEventDestroy(globalStart);
    cudaEventDestroy(globalStop);

#endif // FINE_GRAINED_TIMING

    delete imVol;
    delete doseVol;
    cudaDeviceReset();
}
