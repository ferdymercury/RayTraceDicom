#include <iostream>
#include <limits>

#include "kernel_wrapper.cuh"
#include "energy_reader.h"
#include "float3_affine_transform.cuh"
#include "float3_from_fan_transform.cuh"
#include "float3_to_fan_transform.cuh"
#include "helper_math.h"
#include "vector_find.h"
#include "vector_interpolate.h"
#include "itkImage.h"
#include "itkImageSeriesReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"

int main()
{
    typedef unsigned int uint;
    std::vector<float> imageData;
    const std::string dataPath(PHYS_DATA_DIRECTORY);

    int t;
    t = clock();
    EnergyStruct ciddData = energyReader(dataPath);
    t = clock()-t;

    uint N;
    uint3 dim;
#ifdef WATER_CUBE_TEST
    dim = make_uint3(256, 256, 256);
    N = dim.x*dim.y*dim.z;
    imageData.resize(N, 1000.0f);
    Float3AffineTransform imIdxToWorld(Matrix3x3(1.0f, 1.0f, 1.0f), make_float3(-128.0f, -128.0f, -256.0f+150.0f));

#else // WATER_CUBE_TEST

    typedef signed short PixelType;
    typedef itk::Image<PixelType, 3> ImageType;
    typedef itk::ImageSeriesReader<ImageType> ReaderType;
    typedef itk::MetaDataDictionary DictionaryType;
    typedef itk::GDCMImageIO ImageIOType;
    typedef itk::GDCMSeriesFileNames NamesGeneratorType;
    std::string imagePath(IMG_DATA_DIRECTORY);

    Matrix3x3 imSpacing(0.);
    Matrix3x3 imDir(0.);
    float3 imOrigin;

    //ReaderType::Pointer reader = ReaderType::New();
    ImageIOType::Pointer dicomIO = ImageIOType::New();
    //reader->SetImageIO(dicomIO);
    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
    nameGenerator->SetUseSeriesDetails(true);
    nameGenerator->AddSeriesRestriction("0008|0021");
    nameGenerator->SetDirectory(imagePath);

    try {
        std::cout << "The directory " << imagePath << " contains the following DICOM Series:\n";

        typedef std::vector<std::string> SeriesIdContainer;
        const SeriesIdContainer &seriesUID = nameGenerator->GetSeriesUIDs();

        SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
        SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
        while( seriesItr != seriesEnd ) {
            std::cout << seriesItr->c_str() << std::endl << std::endl;
            ++seriesItr;
        }

        std::string seriesIdentifier;
        seriesIdentifier = seriesUID.begin()->c_str();

        std::cout << "Please wait while reading series:\n";
        std::cout << seriesIdentifier << std::endl;
        t = clock();

        typedef std::vector<std::string> FileNamesContainer;
        FileNamesContainer fileNames;

        fileNames = nameGenerator->GetFileNames(seriesIdentifier);

        //reader->SetFileNames(fileNames);

        try {
            //reader->Update();
        }
        catch (itk::ExceptionObject &ex) {
            std::cout << ex << std::endl;
            std::cin.get();
            return EXIT_FAILURE;
        }
        //const DictionaryType &dictionary = dicomIO->GetMetaDataDictionary();
        //DictionaryType::ConstIterator itr = dictionary.Begin();
        //DictionaryType::ConstIterator end = dictionary.End();

        //while( itr != end ) {
        //  std::cout << itr->first << " " << itr->second << '\n';
        //  itr++;
        //}

        t = clock()-t;
        std::cout << "Done!\n\nRead image: " << (float)t/CLOCKS_PER_SEC << " seconds\n\n";
        /*
        //ImageType *outputImagePtr = reader->GetOutput();
        ImageType::Pointer outputImagePtr = reader->GetOutput();
        itk::ImageRegionIterator<ImageType> it(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());

        ImageType::SizeType imageSize = outputImagePtr->GetLargestPossibleRegion().GetSize();
        dim = make_uint3(uint(imageSize[0]), uint(imageSize[1]), uint(imageSize[2]));
        N = dim.x*dim.y*dim.z;
        imageData.resize(N);

        t = clock();

        it.GoToBegin();
        unsigned int idx = 0;
        while(!it.IsAtEnd())
        {
            imageData[idx] = float(it.Get() + 1000); // Image should be in HU+1000, remember?
            ++idx;
            ++it;
        }

        //PixelType *pixelPtr = outputImagePtr->GetBufferPointer();
        //for(int i=0; i<N; i++) {
        //  imageData[i] = std::max<float>(float(pixelPtr[i]+1000)*0.001f,0.0f);
        //  imageData[i] = std::max<float>(float(pixelPtr[i]), -1000.0f);
        //}
        t = clock()-t;
        std::cout << "Convert image to float (CPU): " << (float)t/CLOCKS_PER_SEC << " seconds.\n\n";

        //t = clock();
        //EnergyStruct energyData = energyReader(energyPath);
        //t = clock()-t;
        //std::cout << "Read energy matrix: " << (float)t/CLOCKS_PER_SEC << " seconds.\n\n";

        itk::Matrix<double,3,3> itkImDir = outputImagePtr->GetDirection();
        itk::Vector<double,3> itkImSpacing = outputImagePtr->GetSpacing();
        itk::Point<double,3> itkImOrigin = outputImagePtr->GetOrigin();
        imDir = Matrix3x3(make_float3(itkImDir[0][0],itkImDir[0][1],itkImDir[0][2]), make_float3(itkImDir[1][0],itkImDir[1][1],itkImDir[1][2]), make_float3(itkImDir[2][0],itkImDir[2][1],itkImDir[2][2]));
        imSpacing = Matrix3x3(make_float3(itkImSpacing.GetDataPointer()[0],itkImSpacing.GetDataPointer()[1],itkImSpacing.GetDataPointer()[2]));
        imOrigin = make_float3(itkImOrigin.GetDataPointer()[0],itkImOrigin.GetDataPointer()[1],itkImOrigin.GetDataPointer()[2]);
        */
    }
    catch (itk::ExceptionObject &ex) {
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }
    Float3AffineTransform imIdxToWorld(imDir*imSpacing, imOrigin);

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
    HostPinnedImage3D<float> doseVol(&doseData[0], dim);
    HostPinnedImage3D<float> imVol(&imageData[0], dim);

    const uint nLayers = 20;
    const uint3 beamDim = make_uint3(33, 33, nLayers);
    const int beamN = beamDim.x*beamDim.y*beamDim.z;
    std::vector<float> beamData(beamN);
    for (int i=0; i<beamN; ++i) {
        beamData[i] = 90.0f + 10.0f * float(rand())/float(RAND_MAX);
    }
    //beamData[0] = 100.0f;
    HostPinnedImage3D<float> spotWeights(&beamData[0], beamDim);
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


    std::cout << "Read cumulative energy matrix: " << (float)t/CLOCKS_PER_SEC << " seconds.\n\n";

    std::cout << "Executing code on GPU...\n\n";

    cudaWrapperProtons(imVol, doseVol, beams, ciddData, std::cout);

    std::cout << "Done!\n\n";

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
