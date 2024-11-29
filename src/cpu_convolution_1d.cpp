/**
 * \file
 * \brief CPU 1D (2D linearized) Convolution function implementations
 */
#include "cpu_convolution_1d.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "constants.h"

//float erf(float x)
//{
//    // Constants
//    double a1 =  0.254829592;
//    double a2 = -0.284496736;
//    double a3 =  1.421413741;
//    double a4 = -1.453152027;
//    double a5 =  1.061405429;
//    double p  =  0.3275911;
//
//    // Save the sign of x
//    int sign = 1;
//    if (x < 0)
//        sign = -1;
//    x = fabs(x);
//
//    // A&S formula 7.1.26
//    double t = 1.0/(1.0 + p*x);
//    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
//
//    return float(sign*y);
//}

void xConvCpu(const float* const in, float* const out, const float rSigmaEff, const unsigned int rad, const unsigned int inWidth, const unsigned int outWidth, const unsigned int height, const int inOutOffset)
{
    std::vector<float> erfDiffs(rad+1);
    float erfNew = std::erf(rSigmaEff*HALF);
    float erfOld;
    erfDiffs[0] = erfNew;
    for (unsigned int i=1; i<rad+1; ++i) {
        erfOld = erfNew;
        erfNew = std::erf(rSigmaEff*(float(i)+HALF));
        erfDiffs[i] = HALF*(erfNew - erfOld);
    }
    for (unsigned int y=0; y<height; ++y) {
        unsigned int yOffsetOut = y*outWidth;
        unsigned int yOffsetIn = y*inWidth;
        for (unsigned int xOut=0; xOut<outWidth; ++xOut) {
            float res = 0.0;
            for (long i=-static_cast<long>(rad); i<rad+1; ++i) {
                long xIn = xOut - inOutOffset + i;
                if (xIn>=0 && xIn<inWidth) {
                    res += erfDiffs[static_cast<size_t>(std::abs(i))] * in[yOffsetIn + xIn];
                }
            }
            out[yOffsetOut + xOut] = res;
        }
    }
}

void xConvCpuScat(const float* const in, float* const out, const float rSigmaEff, const unsigned int rad, const unsigned int inWidth, const unsigned int outWidth, const unsigned int height, const unsigned int inOutOffset)
{
    if (rad > inOutOffset) {
        std::cout << "Error: rad > inOutOffset in xConvCpuScat" << std::endl;
        exit (1);
    }
    std::vector<float> erfDiffs(rad+1);
    float erfNew = std::erf(rSigmaEff*HALF);
    float erfOld;
    erfDiffs[0] = erfNew;
    for (unsigned int i=1; i<rad+1; ++i) {
        erfOld = erfNew;
        erfNew = std::erf(rSigmaEff*(float(i)+HALF));
        erfDiffs[i] = HALF*(erfNew - erfOld);
    }
    for (unsigned int y=0; y<height; ++y) {
        unsigned int yOffsetOut = y*outWidth;
        unsigned int yOffsetIn = y*inWidth;
        for (unsigned int xIn=0; xIn<inWidth; ++xIn) {
            float val = in[yOffsetIn + xIn];
            for (long i=-static_cast<long>(rad); i<rad+1; ++i) {
                long xOut = xIn + inOutOffset + i;
                out[yOffsetOut + xOut] += erfDiffs[static_cast<size_t>(std::abs(i))] * val;
            }
        }
    }
}

void xConvCpuSparse(const float* const in, float* const out, const float rSigmaEff, const unsigned int rad, const unsigned int inWidth, const unsigned int outWidth, const unsigned int height, const int inOutOffset, const int inOutDelta)
{
    if (static_cast<int>(rad) > inOutOffset) {
        std::cout << "Error: rad > inOutOffset in xConvCpuSparse" << std::endl;
        exit (1);
    }
    std::vector<float> erfDiffs(rad+1);
    float erfNew = std::erf(rSigmaEff*HALF);
    float erfOld;
    erfDiffs[0] = erfNew;
    for (unsigned int i=1; i<rad+1; ++i) {
        erfOld = erfNew;
        erfNew = std::erf(rSigmaEff*(float(i)+HALF));
        erfDiffs[i] = HALF*(erfNew - erfOld);
    }
    for (unsigned int y=0; y<height; ++y) {
        unsigned int yOffsetOut = y*outWidth;
        unsigned int yOffsetIn = y*inWidth;
        for (unsigned int xIn=0; xIn<inWidth; ++xIn) {
            float val = in[yOffsetIn + xIn];
            for (long i=-static_cast<long>(rad); i<rad+1; ++i) {
                long xOut = xIn * inOutDelta + inOutOffset + i;
                out[yOffsetOut + xOut] += erfDiffs[static_cast<size_t>(std::abs(i))] * val;
            }
        }
    }
}

//void yConvCpu(float* const in, float* const out, const float rSigmaEff, const int rad, const int inHeight, const int width, const int outHeight, const int inOutOffset)
//{
//  std::vector<float> erfDiffs(rad+1);
//  float erfNew = erf(rSigmaEff*HALF);
//  float erfOld;
//  erfDiffs[0] = erfNew;
//  for (int i=1; i<rad+1; ++i) {
//      erfOld = erfNew;
//      erfNew = erf(rSigmaEff*(float(i)+HALF));
//      erfDiffs[i] = HALF*(erfNew - erfOld);
//  }
//  for (int yOut=0; yOut<outHeight; ++yOut) {
//      for (int i=-rad; i<rad+1; ++i) {
//          int yIn = yOut - inOutOffset + i;
//          if (yIn>=0 && yIn<inHeight) {
//              float erfDiff = erfDiffs[std::abs(i)];
//              int yOutOffset = yOut*width;
//              int yInOffset = yIn*width;
//              for (int x=0; x<width; ++x) {
//                  out[yOutOffset + x] += erfDiff * in[yInOffset + x];
//              }
//          }
//      }
//  }
//}

void yConvCpu(const float* const in, float* const out, const float rSigmaEff, const unsigned int rad, const unsigned int inHeight, const unsigned int width, const int inOutOffset)
{
    if (static_cast<int>(rad) > inOutOffset) {
        std::cout << "Error: rad > inOutOffset in yConvCpu" << std::endl;
        exit (1);
    }
    std::vector<float> erfDiffs(rad+1);
    float erfNew = std::erf(rSigmaEff*HALF);
    float erfOld;
    erfDiffs[0] = erfNew;
    for (unsigned int i=1; i<rad+1; ++i) {
        erfOld = erfNew;
        erfNew = std::erf(rSigmaEff*(float(i)+HALF));
        erfDiffs[i] = HALF*(erfNew - erfOld);
    }
    for (unsigned int yIn=0; yIn<inHeight; ++yIn) {
        for (long i=-static_cast<long>(rad); i<rad+1; ++i) {
            long yOut = yIn + inOutOffset + i;
            float erfDiff = erfDiffs[static_cast<size_t>(std::abs(i))];
            long yOutOffset = yOut*width;
            long yInOffset = yIn*width;
            for (unsigned int x=0; x<width; ++x) {
                out[yOutOffset + x] += erfDiff * in[yInOffset + x];
            }
        }
    }
}

void yConvCpuSparse(const float* const in, float* const out, const float rSigmaEff, const unsigned int rad, const unsigned int inHeight, const unsigned int width, const int inOutOffset, const int inOutDelta)
{
    if (static_cast<int>(rad) > inOutOffset) {
        std::cout << "Error: rad > inOutOffset in yConvCpuSparse" << std::endl;
        exit (1);
    }
    std::vector<float> erfDiffs(rad+1);
    float erfNew = std::erf(rSigmaEff*HALF);
    float erfOld;
    erfDiffs[0] = erfNew;
    for (unsigned int i=1; i<rad+1; ++i) {
        erfOld = erfNew;
        erfNew = std::erf(rSigmaEff*(float(i)+HALF));
        erfDiffs[i] = HALF*(erfNew - erfOld);
    }
    for (unsigned int yIn=0; yIn<inHeight; ++yIn) {
        for (long i=-static_cast<long>(rad); i<rad+1; ++i) {
            long yOut = yIn * inOutDelta + inOutOffset + i;
            float erfDiff = erfDiffs[static_cast<size_t>(std::abs(i))];
            long yOutOffset = yOut*width;
            long yInOffset = yIn*width;
            for (unsigned int x=0; x<width; ++x) {
                out[yOutOffset + x] += erfDiff * in[yInOffset + x];
            }
        }
    }
}
