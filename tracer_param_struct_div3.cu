/**
 * \file
 * \brief TracerParamStructDiv3 class declaration
 */
#include "tracer_param_struct_div3.cuh"
#include "float3_from_fan_transform.cuh"
#include "helper_math.h"
#include "helper_float3.cuh"

TracerParamStructDiv3::TracerParamStructDiv3(const Float3FromFanTransform fanIdxToImIdx)
{
    float2 d2 = fanIdxToImIdx.getSourceDist();
    dist = sqrt(d2.x*d2.x+d2.y*d2.y);///< @todo is this right??
    float3 fanIdxMin = fanIdxToImIdx.getFanIdxToFan().getOffset();
    float3 fanIdxDelta = fanIdxToImIdx.getFanIdxToFan().getDelta();
    Matrix3x3 tTransp = fanIdxToImIdx.getGantryToImIdx().getMatrix().transpose();
    coefOffset = (tTransp.r0*fanIdxMin.x + tTransp.r1*fanIdxMin.y)/dist + tTransp.r2;
    coefIdxI = tTransp.r0*fanIdxDelta.x/dist;
    coefIdxJ = tTransp.r1*fanIdxDelta.y/dist;
    transl = fanIdxToImIdx.getGantryToImIdx().getOffset() - dist*tTransp.r2;

    // Test code, delete!
    print_float3(fanIdxToImIdx.getGantryToImIdx().getOffset());
    print_float3(tTransp.r2);

    minDist = fanIdxMin.z + dist;
    deltaZ = fanIdxDelta.z;

    float mx = fanIdxDelta.x*minDist/dist;
    float my = fanIdxDelta.y*minDist/dist;
    float dx = fanIdxDelta.x*deltaZ/dist;
    float dy = fanIdxDelta.y*deltaZ/dist;
    // dV/dz element for layer at step k given by (volConst + k*volLin + k*k*volSq)
    volConst = 1.0f/3.0f*(dx*my*minDist/deltaZ + dy*mx*minDist/deltaZ + mx*my + 0.25f*dx*dy);
    volLin = 2.0f/3.0f*(dx*dy*minDist/deltaZ + dx*my + dy*mx);
    volSq = dx*dy;
}

//TracerParamStructDiv3::TracerParamStructDiv3(const Float3IdxTransform fanIdxToFan, const float sourceDist, const Float3AffineTransform gantryToWorldIdx)
//{
//  dist = sourceDist;
//  float3 fanIdxMin = fanIdxToFan.getOffset();
//  float3 fanIdxDelta = fanIdxToFan.getDelta();
//  Matrix3x3 tTransp = gantryToWorldIdx.getMatrix().transpose();
//  coefOffset = (tTransp.r0*fanIdxMin.x + tTransp.r1*fanIdxMin.y)/dist + tTransp.r2;
//  coefIdxI = tTransp.r0*fanIdxDelta.x/dist;
//  coefIdxJ = tTransp.r1*fanIdxDelta.y/dist;
//  transl = gantryToWorldIdx.getOffset() - dist*tTransp.r2;
//  minDist = fanIdxMin.z + dist;
//  deltaZ = fanIdxDelta.z;
//
//  float mx = fanIdxDelta.x*minDist/dist;
//  float my = fanIdxDelta.y*minDist/dist;
//  float dx = fanIdxDelta.x*deltaZ/dist;
//  float dy = fanIdxDelta.y*deltaZ/dist;
//  // dV/dz element for layer at step k given by (volConst + k*volLin + k*k*volSq)
//  volConst = 1.0f/3.0f*(dx*my*minDist/deltaZ + dy*mx*minDist/deltaZ + mx*my + 0.25f*dx*dy);
//  volLin = 2.0f/3.0f*(dx*dy*minDist/deltaZ + dx*my + dy*mx);
//  volSq = dx*dy;
//}

//TracerParamStructDiv3::TracerParamStructDiv3(const Float3IdxTransform fanIdxToFan, const float sourceDist, const Float3AffineTransform gantryToWorldIdx)
//{
//  dist = sourceDist;
//  float3 fanIdxMin = fanIdxToFan.getOffset();
//  float3 fanIdxDelta = fanIdxToFan.getDelta();
//  Matrix3x3 tTransp = (gantryToWorldIdx.getMatrix()).transpose();
//  coefOffset = (tTransp.r0*fanIdxMin.x + tTransp.r1*fanIdxMin.y)/sourceDist + tTransp.r2;
//  coefIdxI = tTransp.r0*fanIdxDelta.x/sourceDist;
//  coefIdxJ = tTransp.r1*fanIdxDelta.y/sourceDist;
//  transl = gantryToWorldIdx.getOffset();
//  minDist = fanIdxMin.z;
//  deltaZ = fanIdxDelta.z;
//
//  float mx = fanIdxDelta.x*minDist/dist;
//  float my = fanIdxDelta.y*minDist/dist;
//  float dx = fanIdxDelta.x*deltaZ/dist;
//  float dy = fanIdxDelta.y*deltaZ/dist;
//  // dV/dz element for layer at step k given by (volConst + k*volLin + k*k*volSq)
//  volConst = 1.0f/3.0f*(dx*my*minDist/deltaZ + dy*mx*minDist/deltaZ + mx*my + 0.25f*dx*dy);
//  volLin = 2.0f/3.0f*(dx*dy*minDist/deltaZ + dx*my + dy*mx);
//  volSq = dx*dy;
//}

//TracerParamStructDiv3::TracerParamStructDiv3(const Float3FromFanTransform fanIdxToImIdx)
//{
//  dist = fanIdxToImIdx.getSourceDist();
//  float3 fanIdxMin = fanIdxToImIdx.getFanIdxToFan().getOffset();
//  float3 fanIdxDelta = fanIdxToImIdx.getFanIdxToFan().getDelta();
//  Matrix3x3 tTransp = fanIdxToImIdx.getGantryToImIdx().getMatrix().transpose();
//  coefOffset = (tTransp.r0*fanIdxMin.x + tTransp.r1*fanIdxMin.y)/dist + tTransp.r2;
//  coefIdxI = tTransp.r0*fanIdxDelta.x/dist;
//  coefIdxJ = tTransp.r1*fanIdxDelta.y/dist;
//  transl = fanIdxToImIdx.getGantryToImIdx().getOffset();
//  minDist = fanIdxMin.z;
//  deltaZ = fanIdxDelta.z;
//
//  float mx = fanIdxDelta.x*minDist/dist;
//  float my = fanIdxDelta.y*minDist/dist;
//  float dx = fanIdxDelta.x*deltaZ/dist;
//  float dy = fanIdxDelta.y*deltaZ/dist;
//  // dV/dz element for layer at step k given by (volConst + k*volLin + k*k*volSq)
//  volConst = 1.0f/3.0f*(dx*my*minDist/deltaZ + dy*mx*minDist/deltaZ + mx*my + 0.25f*dx*dy);
//  volLin = 2.0f/3.0f*(dx*dy*minDist/deltaZ + dx*my + dy*mx);
//  volSq = dx*dy;
//}

CUDA_CALLABLE_MEMBER float3 TracerParamStructDiv3::getStart(const int idxI, const int idxJ) const {return (idxI*coefIdxI + idxJ*coefIdxJ + coefOffset)*minDist + transl;}

CUDA_CALLABLE_MEMBER float3 TracerParamStructDiv3::getInc(const int idxI, const int idxJ) const {return (idxI*coefIdxI + idxJ*coefIdxJ + coefOffset)*deltaZ;}

CUDA_CALLABLE_MEMBER float TracerParamStructDiv3::getDeltaZ() const {return deltaZ;}

CUDA_CALLABLE_MEMBER float TracerParamStructDiv3::getMinDist() const {return minDist;}

CUDA_CALLABLE_MEMBER float TracerParamStructDiv3::volPerDist(const int k) const {return volConst + k*volLin + k*k*volSq;}
