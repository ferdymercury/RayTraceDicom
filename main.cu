#include <iostream>
#include <limits>
#include <fstream>

#include "kernel_wrapper.cuh"
#include "energy_reader.h"
#include "float3_affine_transform.cuh"
#include "float3_from_fan_transform.cuh"
#include "float3_to_fan_transform.cuh"
#include "helper_math.h"
#include "vector_find.h"
#include "vector_interpolate.h"
#include "dicom_reader.h"

int main()
{
    typedef unsigned int uint;
    std::vector<float> imageData;
    const std::string dataPath(PHYS_DATA_DIRECTORY);

    clock_t t;
    t = clock();
    EnergyStruct ciddData = energyReader(dataPath);
    t = clock()-t;
    std::cout << "Read energy matrix: " << static_cast<float>(t)/CLOCKS_PER_SEC << " seconds.\n\n";


    uint N;
    uint3 dim;
#ifdef WATER_CUBE_TEST
    dim = make_uint3(256, 256, 256);
    N = dim.x*dim.y*dim.z;
    imageData.resize(N, 1000.0f);
    Float3AffineTransform imIdxToWorld(Matrix3x3(1.0f, 1.0f, 1.0f), make_float3(-128.0f, -128.0f, -256.0f+150.0f));

#else // WATER_CUBE_TEST
    Float3AffineTransform imIdxToWorld = itk_reader(IMG_DATA_DIRECTORY,imageData,N,dim,t);

#endif // WATER_CUBE_TEST

    Float3AffineTransform worldToImIdx(imIdxToWorld.inverse());

    float fInf = std::numeric_limits<float>::infinity();
    float2 sourceDist = make_float2(fInf, fInf);
    //float2 sourceDist = make_float2(1000, 1000);
    Float3AffineTransform worldToGantry(Matrix3x3(1.0f, 1.0f, 1.0f), make_float3(0.0f, 0.0f, 0.0f));
    Float3AffineTransform gantryToWorld = worldToGantry.inverse();
    Float3AffineTransform gantryToImIdx = concatFloat3AffineTransform(gantryToWorld, worldToImIdx);
    ///< @todo Testing, remove!
    //Float3AffineTransform gantryToImIdx = Float3AffineTransform(Matrix3x3(1.326237f, 0.0f, 0.0f, 0.0f, 1.326237f, 0.0f, 0.0f, 0.0f, 0.25f), make_float3(256.147552f, 256.147552f, 51.0f));

    #ifdef WATER_CUBE_TEST
    Float3IdxTransform fanIdxToFan(make_float3(3.0f, 3.0f, -1.0f), make_float3(-48.0f, -48.0f, 128.0f));
    #else // WATER_CUBE_TEST
    Float3IdxTransform fanIdxToFan(make_float3(1.0f, 1.0f, -1.0f), make_float3(0.0f, 0.0f, 104.0f));
    #endif // WATER_CUBE_TEST

    Float3FromFanTransform fanIdxToImIdx(fanIdxToFan, sourceDist, gantryToImIdx);
    Float3ToFanTransform imIdxToFanIdx(gantryToImIdx.inverse(), sourceDist, fanIdxToFan.inverse());

    std::vector<float> doseData(N, 0.0f);
    HostPinnedImage3D<float>* doseVol = new HostPinnedImage3D<float>(&doseData[0], dim);// to be deleted by cudaWrapperProtons, before cudaDeviceReset
    HostPinnedImage3D<float>* imVol = new HostPinnedImage3D<float>(&imageData[0], dim);// to be deleted by cudaWrapperProtons, before cudaDeviceReset

    const uint nLayers = 20;
    const uint3 beamDim = make_uint3(33, 33, nLayers);
    const int beamN = beamDim.x*beamDim.y*beamDim.z;
    std::vector<float> beamData(beamN);
    for (int i=0; i<beamN; ++i) {
        beamData[i] = 90.0f + 10.0f * float(rand())/float(RAND_MAX);
    }
    //beamData[0] = 100.0f;

    HostPinnedImage3D<float>* spotWeights = new HostPinnedImage3D<float>(&beamData[0], beamDim);// to be deleted by cudaWrapperProtons, before cudaDeviceReset

    float currentEnergy = 118.12f;
    float lastEnergy = 172.51f;
    float energyStep = (lastEnergy-currentEnergy) / float(nLayers-1);

    std::vector<float> energiesPerU(nLayers);
    std::vector<float2> sigmas(nLayers);
    for (uint i = 0; i<nLayers; ++i) {
        energiesPerU[i] = currentEnergy;
        float energyIdx = findDecimalOrdered<float, float> (ciddData.energiesPerU, currentEnergy);
        float peakDepth = vectorInterpolate<float,float> (ciddData.peakDepths, energyIdx);
        sigmas[i].x = 2.3f + 290.0f/(peakDepth+15.0f); // Empirical fit
        sigmas[i].y = 2.3f + 290.0f/(peakDepth+15.0f); // Empirical fit
        currentEnergy += energyStep;
    }
    //const uint3 energyDim = make_uint3(nLayers, 1, 1);
    //HostPinnedImage3D<float> beamEnergies(&energiesPerU[0], energyDim);

    unsigned int tracerSteps = 512; // Number of steps traced

    ///< @todo: change to have fITDI different from fITII
    ///< @todo: Testing, change to real vector argument
    std::vector<BeamSettings> beams;
    beams.push_back(BeamSettings(spotWeights, energiesPerU, sigmas, make_float2(1.0f, 1.0f), tracerSteps, sourceDist, fanIdxToFan, gantryToImIdx, gantryToImIdx));

    //std::cout << doseData[512*512*25 + 512*275 + 275] << '\n';


    std::cout << "Read cumulative energy matrix: " << static_cast<float>(t)/CLOCKS_PER_SEC << " seconds.\n\n";

    std::cout << "Executing code on GPU...\n\n";

    cudaWrapperProtons(imVol, doseVol, beams, ciddData, std::cout);

    std::cout << "Done!\n\n";

    //Export dose result to a binary file, that you can open with Amide
    std::ofstream fout("/tmp/dose.dat", std::ios::out | std::ios::binary);
    fout.write(reinterpret_cast<const char*>(&doseData[0]), doseData.size()*sizeof(float));
    fout.close();
    std::cout << "Written /tmp/dose.dat with size " << dim.x << "x" << dim.y << "x" << dim.z <<  "\n\n";

    //std::cout << doseData[512*512*25 + 512*275 + 275] << '\n';

    //cudaWrapper(&imageData[0], dim, fanIdxToImIdx, cumulEnergyData);

    //TracerParamStructDiv3 testParams(fanIdxToImIdx);
    //std::cout << testParams.relStepLen(0, 0) << " " << testParams.relStepLen(5000, 5000) << " " << sqrt(3.0) << '\n';

    //int testI = 17;
    //int testJ = 7;
    //float3 pos = paramsDiv.getStart(testI, testJ);
    //float3 inc = paramsDiv.getInc(testI, testJ);
    //for(int i=0; i<12; i++) {
    //  printf("%d %d %d\n", testI, testJ, i);
    //  float3 intPos = make_float3(int(pos.x), int(pos.y), int(pos.z));
    //  print_float3(pos);
    //  print_float3(intPos);
    //  testTransfer.init(pos.x, pos.y);
    //  print_float3(imIdxToFanIdx.transformPoint(pos));
    //  print_float3(imIdxToFanIdx.transformPoint(intPos));
    //  print_float3(testTransfer.getFanIdx(pos.z));
    //  printf("\n");
    //  pos += inc;
    //}

    //TracerParamStructDiv3 paramsDiv2(Float3FromFanTransform(Float3IdxTransform(), 1.0f, Float3AffineTransform()));
    //paramsDiv2.createTest(fanIdxToImIdx);

    //print_float3(paramsDiv.getStart(13,17));
    //print_float3(paramsDiv2.getStart2(13,17));
    //print_float3(paramsDiv.getInc(13,17));
    //print_float3(paramsDiv2.getInc2(13,17));
    //printf("vol: %f, %f\n", paramsDiv.volPerDist(14), paramsDiv2.volPerDist(14));

    //std::cin.get();

    return 0;
}
