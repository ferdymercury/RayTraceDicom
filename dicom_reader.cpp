/**
 * \file
 * \brief Function implementation for reading DICOM images
 */

#include "dicom_reader.h"
#include "float3_affine_transform.cuh"
#include "cuda_runtime.h"

#include "itkImage.h"
#include "itkImageSeriesReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"

Float3AffineTransform itk_reader(const std::string& imagePath, std::vector<float>& imageData, unsigned int& N, uint3& dim, clock_t &t)
{
    using PixelType = short;
    using ImageType = itk::Image<PixelType, 3>;
    using ReaderType = itk::ImageSeriesReader<ImageType>;
    using DictionaryType = itk::MetaDataDictionary;
    using ImageIOType = itk::GDCMImageIO;
    using NamesGeneratorType = itk::GDCMSeriesFileNames;

    constexpr short HUOFFSET = 1000;

    Matrix3x3 imSpacing(0.);
    Matrix3x3 imDir(0.);
    float3 imOrigin;

    ReaderType::Pointer reader = ReaderType::New();
    ImageIOType::Pointer dicomIO = ImageIOType::New();
    reader->SetImageIO(dicomIO);
    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
    nameGenerator->SetUseSeriesDetails(true);
    nameGenerator->AddSeriesRestriction("0008|0021");
    nameGenerator->SetDirectory(imagePath);

    try {        
        using SeriesIdContainer = std::vector<std::string>;
        const SeriesIdContainer &seriesUID = nameGenerator->GetSeriesUIDs();

        if(seriesUID.empty())
        {
            std::cout << "The directory " << imagePath << " contains no DICOM Series:\n";
            throw itk::ExceptionObject();
        }

        std::cout << "The directory " << imagePath << " contains the following DICOM Series:\n";
        auto seriesItr = seriesUID.begin();
        auto seriesEnd = seriesUID.end();
        while( seriesItr != seriesEnd ) {
            std::cout << seriesItr->c_str() << std::endl << std::endl;
            ++seriesItr;
        }

        std::string seriesIdentifier;
        seriesIdentifier = *seriesUID.begin();

        std::cout << "Please wait while reading series:\n";
        std::cout << seriesIdentifier << std::endl;
        t = clock();

        using FileNamesContainer = std::vector<std::string>;
        FileNamesContainer fileNames;

        fileNames = nameGenerator->GetFileNames(seriesIdentifier);

        reader->SetFileNames(fileNames);

        try {
            reader->Update();
        }
        catch (itk::ExceptionObject &ex) {
            std::cout << ex << std::endl;
            std::cin.get();
            throw std::runtime_error("Abort due to ITK exception");
        }
        //const DictionaryType &dictionary = dicomIO->GetMetaDataDictionary();
        //DictionaryType::ConstIterator itr = dictionary.Begin();
        //DictionaryType::ConstIterator end = dictionary.End();

        //while( itr != end ) {
        //  std::cout << itr->first << " " << itr->second << '\n';
        //  itr++;
        //}

        t = clock()-t;
        std::cout << "Done!\n\nRead image: " << static_cast<float>(t)/CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

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
            imageData[idx] = float(it.Get() + HUOFFSET); ///\todo Image should be in HU+1000, remember? ///rescale intercept??
            ++idx;
            ++it;
        }

        //PixelType *pixelPtr = outputImagePtr->GetBufferPointer();
        //for(int i=0; i<N; i++) {
        //  imageData[i] = std::max<float>(float(pixelPtr[i]+1000)*0.001f,0.0f);
        //  imageData[i] = std::max<float>(float(pixelPtr[i]), -1000.0f);
        //}
        t = clock()-t;
        std::cout << "Convert image to float (CPU): " << static_cast<float>(t)/CLOCKS_PER_SEC << " seconds." << std::endl << std::endl;

        itk::Matrix<double,3,3> itkImDir = outputImagePtr->GetDirection();
        itk::Vector<double,3> itkImSpacing = outputImagePtr->GetSpacing();
        itk::Point<double,3> itkImOrigin = outputImagePtr->GetOrigin();
        imDir = Matrix3x3(make_float3(itkImDir[0][0],itkImDir[0][1],itkImDir[0][2]), make_float3(itkImDir[1][0],itkImDir[1][1],itkImDir[1][2]), make_float3(itkImDir[2][0],itkImDir[2][1],itkImDir[2][2]));
        imSpacing = Matrix3x3(make_float3(itkImSpacing.GetDataPointer()[0],itkImSpacing.GetDataPointer()[1],itkImSpacing.GetDataPointer()[2]));
        imOrigin = make_float3(itkImOrigin.GetDataPointer()[0],itkImOrigin.GetDataPointer()[1],itkImOrigin.GetDataPointer()[2]);
    }
    catch (itk::ExceptionObject &ex) {
      std::cout << ex << std::endl;
      throw std::runtime_error("Abort due to ITK exception");
    }
    return Float3AffineTransform(imDir*imSpacing, imOrigin);
}
