/**
 * \file
 * \brief TracerParamStruct3 struct declaration
 */
#ifndef TRACER_PARAM_STRUCT3_H
#define TRACER_PARAM_STRUCT3_H

#include "vector_types.h"

/**
 * \brief 3D tracing parameter struct
 */
struct TracerParamStruct3{
    float3 step;            ///< 3D step length
    float3 start;           ///< Start 3D point
    float3 gantryX;         ///< Gantry 3D x axis direction vector
    float3 gantryY;         ///< Gantry 3D y axis direction vector
    unsigned int steps;     ///< Number of steps
    float worldStepLength;  ///< step length in the world coordinate system
};

#endif // TRACER_PARAM_STRUCT3_H
