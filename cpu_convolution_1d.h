/**
 * \file
 * \brief CPU 1D (2D linearized) Convolution function declarations
 */
#ifndef CPU_CONVOLUTION_1D_H
#define CPU_CONVOLUTION_1D_H

/**
 * \brief Gaussian convolution of a 2D image in x
 * \param in pointer to 2D input data array (not owned), linearized
 * \param out pointer to 2D output data array (linearized) where convolution result will be stored (not owned, preallocated by user)
 * \param rSigmaEff Gaussian convolution sigma
 * \param rad convolution radius cutoff (half-width)
 * \param inWidth width of the 2D matrix "in"
 * \param outWidth width of the 2D matrix "out"
 * \param outHeight height of the 2D matrix "out"
 * \param inOutOffset shift of the output vs input
 * \return void
 */
void xConvCpu(const float* const in, float* const out, const float rSigmaEff, const unsigned int rad, const unsigned int inWidth, const unsigned int outWidth, const unsigned int outHeight, const int inOutOffset);

/**
 * \brief Gaussian/scatter convolution of a 2D image in x
 * \param in pointer to 2D input data array (not owned), linearized
 * \param out pointer to 2D output data array (linearized) where convolution result will be stored (not owned, preallocated by user)
 * \param rSigmaEff Gaussian convolution sigma
 * \param rad convolution radius cutoff (half-width)
 * \param inWidth width of the 2D matrix "in"
 * \param outWidth width of the 2D matrix "out"
 * \param outHeight height of the 2D matrix "out"
 * \param inOutOffset shift of the output vs input
 * \return void
 */
void xConvCpuScat(const float* const in, float* const out, const float rSigmaEff, const unsigned int rad, const unsigned int inWidth, const unsigned int outWidth, const unsigned int outHeight, const unsigned int inOutOffset);

/**
 * \brief Gaussian/scatter sparse convolution of a 2D image in x
 * \param in pointer to 2D input data array (not owned), linearized
 * \param out pointer to 2D output data array (linearized) where convolution result will be stored (not owned, preallocated by user)
 * \param rSigmaEff Gaussian convolution sigma
 * \param rad convolution radius cutoff (half-width)
 * \param inWidth width of the 2D matrix "in"
 * \param outWidth width of the 2D matrix "out"
 * \param outHeight height of the 2D matrix "out"
 * \param inOutOffset shift of the output vs input
 * \param inOutDelta shift of the output vs input in perpendicular direction
 * \return void
 */
void xConvCpuSparse(const float* const in, float* const out, const float rSigmaEff, const unsigned int rad, const unsigned int inWidth, const unsigned int outWidth, const unsigned int outHeight, const int inOutOffset, const int inOutDelta);

/**
 * \brief Gaussian convolution of a 2D image in y
 * \param in pointer to 2D input data array (not owned), linearized
 * \param out pointer to 2D output data array (linearized) where convolution result will be stored (not owned, preallocated by user)
 * \param rSigmaEff Gaussian convolution sigma
 * \param rad convolution radius cutoff (half-width)
 * \param inHeight height of the 2D matrix "in"
 * \param outWidth width of the 2D matrix "out"
 * \param inOutOffset shift of the output vs input
 * \return void
 */
void yConvCpu(const float* const in, float* const out, const float rSigmaEff, const unsigned int rad, const unsigned int inHeight, const unsigned int outWidth, const int inOutOffset);

/**
 * \brief Gaussian/scatter sparse convolution of a 2D image in y
 * \param in pointer to 2D input data array (not owned), linearized
 * \param out pointer to 2D output data array (linearized) where convolution result will be stored (not owned, preallocated by user)
 * \param rSigmaEff Gaussian convolution sigma
 * \param rad convolution radius cutoff (half-width)
 * \param inHeight height of the 2D matrix "in"
 * \param outWidth width of the 2D matrix "out"
 * \param inOutOffset shift of the output vs input
 * \param inOutDelta shift of the output vs input in perpendicular direction
 * \return void
 */
void yConvCpuSparse(const float* const in, float* const out, const float rSigmaEff, const unsigned int rad, const unsigned int inHeight, const unsigned int outWidth, const int inOutOffset, const int inOutDelta);

#endif // CPU_CONVOLUTION_1D_H
