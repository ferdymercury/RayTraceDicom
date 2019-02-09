/**
 * \file
 * \brief Function declarations for reading DICOM images
 */

#ifndef DICOM_READER_H
#define DICOM_READER_H

#include <string>
#include <vector>

class Float3AffineTransform;
struct uint3;

/**
 * \brief Read DICOM files with ITK and store them in a matrix
 * \param imagePath the path where the DICOM files are stored
 * \param imageData where the HU CT matrix will be stored
 * \param N number of voxels in the matrix
 * \param dim number of pixels in each dimension
 * \param t the time clock
 * \return the CT orientation in space as an affine transform
 */
Float3AffineTransform itk_reader(const std::string imagePath, std::vector<float>& imageData, unsigned int& N, uint3& dim, int& t);

#endif // DICOM_READER_H
