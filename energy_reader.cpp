/**
 * \file
 * \brief energyReader function implementation
 */
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>
#include "energy_reader.h"
#include "energy_struct.h"

EnergyStruct energyReader(const std::string dataPath) {

    EnergyStruct eStr;
    std::ifstream fileReader;
    fileReader.exceptions(std::ifstream::failbit);

    try {
        fileReader.open(dataPath + "proton_cumul_ddd_data.txt");
    }
    catch (const std::exception&) {
        std::string msg = "Failed to open " + dataPath + "proton_cumul_ddd_data.txt";
        throw std::runtime_error(msg.c_str());
    }

    fileReader >> eStr.nEnergySamples >> eStr.nEnergies;

    eStr.energiesPerU.resize(eStr.nEnergies);
    for (int i=0; i<eStr.nEnergies; ++i) {
        fileReader >> eStr.energiesPerU[i];
    }

    eStr.peakDepths.resize(eStr.nEnergies);
    for (int i=0; i<eStr.nEnergies; ++i) {
        fileReader >> eStr.peakDepths[i];
    }

    eStr.scaleFacts.resize(eStr.nEnergies);
    for (int i=0; i<eStr.nEnergies; ++i) {
        fileReader >> eStr.scaleFacts[i];
    }

    eStr.ciddMatrix.resize(eStr.nEnergySamples*eStr.nEnergies);
    for (int i=0; i<eStr.nEnergySamples*eStr.nEnergies; ++i) {
        fileReader >> eStr.ciddMatrix[i];
    }
    fileReader.close();

    try {
        fileReader.open(dataPath + "density_Schneider2000_adj.txt");
    }
    catch (const std::runtime_error&) {
        std::string msg = "Failed to open " + dataPath + "density_Schneider2000_adj.txt";
        throw std::runtime_error(msg.c_str());
    }
    fileReader >> eStr.nDensitySamples >> eStr.densityScaleFact;
    eStr.densityVector.resize(eStr.nDensitySamples);
    for (int i=0; i<eStr.nDensitySamples; ++i) {
        fileReader >> eStr.densityVector[i];
    }
    fileReader.close();

    try {
        fileReader.open(dataPath + "HU_to_SP_CNAO_H&N_adj.txt");
    }
    catch (const std::runtime_error&) {
        std::string msg = "Failed to open " + dataPath + "HU_to_SP_CNAO_H&N_adj.txt";
        throw std::runtime_error(msg.c_str());
    }
    fileReader >> eStr.nSpSamples >> eStr.spScaleFact;
    eStr.spVector.resize(eStr.nSpSamples);
    for (int i=0; i<eStr.nSpSamples; ++i) {
        fileReader >> eStr.spVector[i];
    }
    fileReader.close();

#ifdef WATER_CUBE_TEST
    try {
        fileReader.open(dataPath + "radiation_length_inc_water.txt");
    }
    catch (const std::runtime_error& e) {
        std::string msg = "Failed to open " + dataPath + "radiation_length_inc_water.txt";
        throw std::runtime_error(msg.c_str());
    }
#else // WATER_CUBE_TEST
    try {
        fileReader.open(dataPath + "radiation_length.txt");
    }
    catch (const std::runtime_error&) {
        std::string msg = "Failed to open " + dataPath + "radiation_length.txt";
        throw std::runtime_error(msg.c_str());
    }
#endif // WATER_CUBE_TEST

    fileReader >> eStr.nRRlSamples >> eStr.rRlScaleFact;
    eStr.rRlVector.resize(eStr.nRRlSamples);
    //std::cout << eStr.rRlSamples << '\n';
    for (int i=0; i<eStr.nRRlSamples; ++i) {
        fileReader >> eStr.rRlVector[i];
    }
    fileReader.close();

#ifdef NUCLEAR_CORR

    int nSamples, nEnergies;
    float energyPerU, peakDepth, scaleFact;

#if NUCLEAR_CORR == SOUKUP
    std::string nucParamFileName = "nuclear_weights_and_sigmas_Soukup.txt";
#elif NUCLEAR_CORR == FLUKA
    std::string nucParamFileName = "nuclear_weights_and_sigmas_Fluka.txt";
#elif NUCLEAR_CORR == GAUSS_FIT
    std::string nucParamFileName = "nuclear_weights_and_sigmas_fit.txt";
#endif



    try {
        fileReader.open(dataPath + nucParamFileName);
    }
    catch (const std::runtime_error& ) {
        std::string msg = "Failed to open " + dataPath + nucParamFileName;
        throw std::runtime_error(msg.c_str());
    }
    fileReader >> nSamples >> nEnergies;
    if (eStr.nEnergySamples != nSamples || eStr.nEnergies != nEnergies) {
        std::string msg = "Number of samples or energies in " + nucParamFileName + " different from proton_cumul_ddd_data.txt";
        throw std::runtime_error(msg.c_str());
    }
    for (int i=0; i<eStr.nEnergies; ++i) {
        fileReader >> energyPerU;
        if ( abs(eStr.energiesPerU[i]-energyPerU) > 0.01 ) {
            std::string msg = "Energies in " + nucParamFileName + " different from proton_cumul_ddd_data.txt";
            throw std::runtime_error(msg.c_str());
        }
    }
    for (int i=0; i<eStr.nEnergies; ++i) {
        fileReader >> peakDepth;
        if ( abs(eStr.peakDepths[i]-peakDepth) > 0.01 ) {
            std::string msg = "Peak depths in " + nucParamFileName + " different from proton_cumul_ddd_data.txt";
            throw std::runtime_error(msg.c_str());
        }
    }
    for (int i=0; i<eStr.nEnergies; ++i) {
        fileReader >> scaleFact;
        if ( abs(eStr.scaleFacts[i]-scaleFact) > 0.01f ) {
            std::string msg = "Scale facts in " + nucParamFileName + " different from proton_cumul_ddd_data.txt";
            throw std::runtime_error(msg.c_str());
        }
    }
    eStr.nucWeightMatrix.resize(eStr.nEnergySamples*eStr.nEnergies);
    for (int i=0; i<eStr.nEnergySamples*eStr.nEnergies; ++i) {
        fileReader >> eStr.nucWeightMatrix[i];
    }
    eStr.nucSqSigmaMatrix.resize(eStr.nEnergySamples*eStr.nEnergies);
    for (int i=0; i<eStr.nEnergySamples*eStr.nEnergies; ++i) {
        fileReader >> eStr.nucSqSigmaMatrix[i];
    }

    fileReader.close();

#endif // NUCLEAR_CORR

    return eStr;
}
