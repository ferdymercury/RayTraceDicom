/**
 * \file
 * \brief CPU 1D Convolution function implementations
 */
#include "cpu_convolution_1d.h"
#include <cmath>
#include <vector>
#include <iostream>

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

void xConvCpu(float* const in, float* const out, const float rSigmaEff, const int rad, const int inWidth, const int outWidth, const int height, const int inOutOffset)
{
    std::vector<float> erfDiffs(rad+1);
    float erfNew = erf(rSigmaEff*0.5f);
    float erfOld;
    erfDiffs[0] = erfNew;
    for (int i=1; i<rad+1; ++i) {
        erfOld = erfNew;
        erfNew = erf(rSigmaEff*(float(i)+0.5f));
        erfDiffs[i] = 0.5f*(erfNew - erfOld);
    }
    for (int y=0; y<height; ++y) {
        int yOffsetOut = y*outWidth;
        int yOffsetIn = y*inWidth;
        for (int xOut=0; xOut<outWidth; ++xOut) {
            float res = 0.0f;
            for (int i=-rad; i<rad+1; ++i) {
                int xIn = xOut - inOutOffset + i;
                if (xIn>=0 && xIn<inWidth) {
                    res += erfDiffs[abs(i)] * in[yOffsetIn + xIn];
                }
            }
            out[yOffsetOut + xOut] = res;
        }
    }
}

void xConvCpuScat(float* const in, float* const out, const float rSigmaEff, const int rad, const int inWidth, const int outWidth, const int height, const int inOutOffset)
{
    if (rad > inOutOffset) {
        printf("Error: rad > inOutOffset in xConvCpuScat");
        exit (1);
    }
    std::vector<float> erfDiffs(rad+1);
    float erfNew = erf(rSigmaEff*0.5f);
    float erfOld;
    erfDiffs[0] = erfNew;
    for (int i=1; i<rad+1; ++i) {
        erfOld = erfNew;
        erfNew = erf(rSigmaEff*(float(i)+0.5f));
        erfDiffs[i] = 0.5f*(erfNew - erfOld);
    }
    for (int y=0; y<height; ++y) {
        int yOffsetOut = y*outWidth;
        int yOffsetIn = y*inWidth;
        for (int xIn=0; xIn<inWidth; ++xIn) {
            float val = in[yOffsetIn + xIn];
            for (int i=-rad; i<rad+1; ++i) {
                int xOut = xIn + inOutOffset + i;
                out[yOffsetOut + xOut] += erfDiffs[abs(i)] * val;
            }
        }
    }
}

void xConvCpuSparse(float* const in, float* const out, const float rSigmaEff, const int rad, const int inWidth, const int outWidth, const int height, const int inOutOffset, const int inOutDelta)
{
    if (rad > inOutOffset) {
        printf("Error: rad > inOutOffset in xConvCpuSparse");
        exit (1);
    }
    std::vector<float> erfDiffs(rad+1);
    float erfNew = erf(rSigmaEff*0.5f);
    float erfOld;
    erfDiffs[0] = erfNew;
    for (int i=1; i<rad+1; ++i) {
        erfOld = erfNew;
        erfNew = erf(rSigmaEff*(float(i)+0.5f));
        erfDiffs[i] = 0.5f*(erfNew - erfOld);
    }
    for (int y=0; y<height; ++y) {
        int yOffsetOut = y*outWidth;
        int yOffsetIn = y*inWidth;
        for (int xIn=0; xIn<inWidth; ++xIn) {
            float val = in[yOffsetIn + xIn];
            for (int i=-rad; i<rad+1; ++i) {
                int xOut = xIn * inOutDelta + inOutOffset + i;
                out[yOffsetOut + xOut] += erfDiffs[abs(i)] * val;
            }
        }
    }
}

//void yConvCpu(float* const in, float* const out, const float rSigmaEff, const int rad, const int inHeight, const int width, const int outHeight, const int inOutOffset)
//{
//  std::vector<float> erfDiffs(rad+1);
//  float erfNew = erf(rSigmaEff*0.5f);
//  float erfOld;
//  erfDiffs[0] = erfNew;
//  for (int i=1; i<rad+1; ++i) {
//      erfOld = erfNew;
//      erfNew = erf(rSigmaEff*(float(i)+0.5f));
//      erfDiffs[i] = 0.5f*(erfNew - erfOld);
//  }
//  for (int yOut=0; yOut<outHeight; ++yOut) {
//      for (int i=-rad; i<rad+1; ++i) {
//          int yIn = yOut - inOutOffset + i;
//          if (yIn>=0 && yIn<inHeight) {
//              float erfDiff = erfDiffs[abs(i)];
//              int yOutOffset = yOut*width;
//              int yInOffset = yIn*width;
//              for (int x=0; x<width; ++x) {
//                  out[yOutOffset + x] += erfDiff * in[yInOffset + x];
//              }
//          }
//      }
//  }
//}

void yConvCpu(float* const in, float* const out, const float rSigmaEff, const int rad, const int inHeight, const int width, const int outHeight, const int inOutOffset)
{
    if (rad > inOutOffset) {
        printf("Error: rad > inOutOffset in yConvCpu");
        exit (1);
    }
    std::vector<float> erfDiffs(rad+1);
    float erfNew = erf(rSigmaEff*0.5f);
    float erfOld;
    erfDiffs[0] = erfNew;
    for (int i=1; i<rad+1; ++i) {
        erfOld = erfNew;
        erfNew = erf(rSigmaEff*(float(i)+0.5f));
        erfDiffs[i] = 0.5f*(erfNew - erfOld);
    }
    for (int yIn=0; yIn<inHeight; ++yIn) {
        for (int i=-rad; i<rad+1; ++i) {
            int yOut = yIn + inOutOffset + i;
            float erfDiff = erfDiffs[abs(i)];
            int yOutOffset = yOut*width;
            int yInOffset = yIn*width;
            for (int x=0; x<width; ++x) {
                out[yOutOffset + x] += erfDiff * in[yInOffset + x];
            }
        }
    }
}

void yConvCpuSparse(float* const in, float* const out, const float rSigmaEff, const int rad, const int inHeight, const int width, const int outHeight, const int inOutOffset, const int inOutDelta)
{
    if (rad > inOutOffset) {
        printf("Error: rad > inOutOffset in yConvCpuSparse");
        exit (1);
    }
    std::vector<float> erfDiffs(rad+1);
    float erfNew = erf(rSigmaEff*0.5f);
    float erfOld;
    erfDiffs[0] = erfNew;
    for (int i=1; i<rad+1; ++i) {
        erfOld = erfNew;
        erfNew = erf(rSigmaEff*(float(i)+0.5f));
        erfDiffs[i] = 0.5f*(erfNew - erfOld);
    }
    for (int yIn=0; yIn<inHeight; ++yIn) {
        for (int i=-rad; i<rad+1; ++i) {
            int yOut = yIn * inOutDelta + inOutOffset + i;
            float erfDiff = erfDiffs[abs(i)];
            int yOutOffset = yOut*width;
            int yInOffset = yIn*width;
            for (int x=0; x<width; ++x) {
                out[yOutOffset + x] += erfDiff * in[yInOffset + x];
            }
        }
    }
}
