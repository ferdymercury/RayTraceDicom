/**
 * \file
 * \brief EnergyStruct declaration
 */
#ifndef ENERGY_STRUCT_H
#define ENERGY_STRUCT_H

#include <vector>

/**
 * \brief ...
 */
struct EnergyStruct{
    int nEnergySamples = 0;             ///< number of energy bins
    int nEnergies = 0;                  ///< number of energies
    std::vector<float> energiesPerU;    ///< energy per bin
    std::vector<float> peakDepths;      ///< proton penetration depth per bin
    std::vector<float> scaleFacts;      ///< scaling factor per bin
    std::vector<float> ciddMatrix;      ///< 2D (lineared) matrix of cumulative integral dose as function of energy and depth?

    int nDensitySamples = 0;            ///< number of density bins
    float densityScaleFact = 0;         ///< density scaling factor
    std::vector<float> densityVector;   ///< densities for each HU

    int nSpSamples = 0;                 ///< number of stopping power bins
    float spScaleFact = 0;              ///< scaling factor for stopping power
    std::vector<float> spVector;        ///< stopping power for each HU

    int nRRlSamples = 0;                ///< number of radiation length bins
    float rRlScaleFact = 0;             ///< radiation length scaling factor
    std::vector<float> rRlVector;       ///< radiation length for each HU

#ifdef NUCLEAR_CORR
    std::vector<float> nucWeightMatrix; ///< nuclear weight correction matrix
    std::vector<float> nucSqSigmaMatrix;///< nuclear sigma matrix?
#endif // NUCLEAR_CORR

};

#endif // ENERGY_STRUCT_H
