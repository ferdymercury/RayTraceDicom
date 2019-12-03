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
#include "config.h"

#include <rti/base/rti_treatment_session.hpp>

int main(int argc, char **argv)
{
    // Read CLI arguments
    const Config config(argc,argv);
    if(config.exit) return 0;

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
    Float3AffineTransform imIdxToWorld = itk_reader(config.ct_dir,imageData,N,dim,t);

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

    #ifdef WATER_CUBE_TEST
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
    #else // WATER_CUBE_TEST


    uint nSpots = 0;
    uint nLayers = 0;

    #ifndef __CUDA_ARCH__
    rti::ct<float> pt(config.ct_dir);
    rti::vec3<float> ct_center = pt.get_center();
    rti::vec3<float> ct_size   = pt.get_size();
    rti::vec3<size_t> ct_nxyz  = pt.get_nxyz();

    const std::unique_ptr<rti::treatment_session<float>> tx_session(new rti::treatment_session<float>(config.rtplan));
    const rti::modality_type m_type = tx_session->get_modality_type();
    const size_t nFields = config.beams.size();
    if(nFields > 1)
    {
        throw std::runtime_error("Multi-beam calculation not yet supported");
    }

    for (size_t i = 0; i < nFields; ++i)
    {
        std::cout << "Loading field " << i << " corresponding to beamname " << config.beams[i] << std::endl;
        auto ds = tx_session->get_beam_dataset(config.beams[i]);
        rti::coordinate_transform<float> p_coord = tx_session -> get_coordinate(config.beams[i]);
        rti::beamline<float>             beam_line = tx_session -> get_beamline(config.beams[i]);
        p_coord.dump();
        for(auto& geo : beam_line.get_geometries())
        {
            geo->dump();
        }
        auto seq_tags = &rti::seqtags_per_modality.at(m_type);
        auto layer0   = (*ds)(seq_tags->at("ctrl"))[0];
        std::array<float,3> angles;
        std::vector<float> tmp;
        layer0->get_values("BeamLimitingDeviceAngle", tmp);
        angles[0] = tmp[0];
        layer0->get_values("GantryAngle", tmp);
        angles[1] = tmp[0];
        layer0->get_values("PatientSupportAngle", tmp);
        angles[2] = tmp[0];

        rti::vec3<float> iso_center(0.0,0.0,0.0);
        if(m_type != rti::RTPLAN && m_type != rti::IONPLAN) {
            throw std::runtime_error("Unknown modality");
        }

        layer0->get_values("IsocenterPosition", tmp);
        iso_center.x = tmp[0];
        iso_center.y = tmp[1];
        iso_center.z = tmp[2];
        std::cout << "Angles: " << angles[0] << " " << angles[1] << " " << angles[2] << " deg" << std::endl;
        std::cout << "IsoCenter: " << iso_center.x << " " << iso_center.y << " " << iso_center.z << " mm" << std::endl;
        std::cout << "ImgCenter: " << ct_center.x << " " << ct_center.y << " " << ct_center.z << " mm" << std::endl;
        std::cout << "ImgSize: " << ct_size.x << " " << ct_size.y << " " << ct_size.z << " mm" << std::endl;
        std::cout << "ImgPixels: " << ct_nxyz.x << " " << ct_nxyz.y << " " << ct_nxyz.z << std::endl;
        std::cout << std::endl;

        rti::beam_module_ion ion_beam(ds,m_type);
        nLayers = ion_beam.get_nb_spots_per_layer()->size();
        const std::vector<rti::beam_module_ion::spot>* sequence = ion_beam.get_sequence();
        nSpots = sequence->size();
        uint sacc = 0;
        uint layer = 0;
        for(auto layerSpots : *ion_beam.get_nb_spots_per_layer())
        {
            std::cout << "Layer " << layer << std::endl;
            for(uint s = sacc; s < sacc + layerSpots; s++)
            {
                auto sp = sequence->at(s);
                std::cout<<"Spot " << s << ": (E,X,Y,Sx,Sy,W): "<< sp.e << ", "
                     << sp.x <<", "<< sp.y <<", "
                     << sp.fwhm_x <<", " << sp.fwhm_y << ", "
                     << sp.meterset <<std::endl;
            }
            std::cout << std::endl;

            layer++;
            sacc += layerSpots;
        }
    }
    #endif // __CUDA_ARCH__

    std::vector<float> beamData(nSpots);
    HostPinnedImage3D<float>* spotWeights = new HostPinnedImage3D<float>(&beamData[0], make_uint3(1,1,nSpots));// to be deleted by cudaWrapperProtons, before cudaDeviceReset
    std::vector<float> energiesPerU(nLayers);
    std::vector<float2> sigmas(nLayers);

    #endif // WATER_CUBE_TEST

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
    std::ofstream fout((config.output_directory+"/dose.dat").c_str(), std::ios::out | std::ios::binary);
    fout.write(reinterpret_cast<const char*>(&doseData[0]), doseData.size()*sizeof(float));
    fout.close();
    std::cout << "Written " << config.output_directory << "/dose.dat with size " << dim.x << "x" << dim.y << "x" << dim.z <<  "\n\n";

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
