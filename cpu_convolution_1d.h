/**
 * \file
 * \brief CPU 1D Convolution function declarations
 */
#ifndef CPU_CONVOLUTION_1D_H
#define CPU_CONVOLUTION_1D_H

/**
 * \brief ...
 * \param in ...
 * \param out ...
 * \param rSigmaEff ...
 * \param rad ...
 * \param inWidth ...
 * \param outWidth ...
 * \param outHeight ...
 * \param inOutOffset ...
 * \return void
 */
void xConvCpu(float* const in, float* const out, const float rSigmaEff, const int rad, const int inWidth, const int outWidth, const int outHeight, const int inOutOffset);

/**
 * \brief ...
 * \param in ...
 * \param out ...
 * \param rSigmaEff ...
 * \param rad ...
 * \param inWidth ...
 * \param outWidth ...
 * \param outHeight ...
 * \param inOutOffset ...
 * \return void
 */
void xConvCpuScat(float* const in, float* const out, const float rSigmaEff, const int rad, const int inWidth, const int outWidth, const int outHeight, const int inOutOffset);

/**
 * \brief ...
 * \param in ...
 * \param out ...
 * \param rSigmaEff ...
 * \param rad ...
 * \param inWidth ...
 * \param outWidth ...
 * \param outHeight ...
 * \param inOutOffset ...
 * \param inOutDelta ...
 * \return void
 */
void xConvCpuSparse(float* const in, float* const out, const float rSigmaEff, const int rad, const int inWidth, const int outWidth, const int outHeight, const int inOutOffset, const int inOutDelta);

/**
 * \brief ...
 * \param in ...
 * \param out ...
 * \param rSigmaEff ...
 * \param rad ...
 * \param inHeight ...
 * \param outWidth ...
 * \param outHeight ...
 * \param inOutOffset ...
 * \return void
 */
void yConvCpu(float* const in, float* const out, const float rSigmaEff, const int rad, const int inHeight, const int outWidth, const int outHeight, const int inOutOffset);

/**
 * \brief ...
 * \param in ...
 * \param out ...
 * \param rSigmaEff ...
 * \param rad ...
 * \param inHeight ...
 * \param outWidth ...
 * \param outHeight ...
 * \param inOutOffset ...
 * \param inOutDelta ...
 * \return void
 */
void yConvCpuSparse(float* const in, float* const out, const float rSigmaEff, const int rad, const int inHeight, const int outWidth, const int outHeight, const int inOutOffset, const int inOutDelta);

#endif // CPU_CONVOLUTION_1D_H