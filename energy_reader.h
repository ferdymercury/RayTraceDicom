/**
 * \file
 * \brief energyReader function declaration
 */
#ifndef ENERGY_READER_H
#define ENERGY_READER_H

struct EnergyStruct;
#include <string>

/**
 * \brief Reads a data file proton_cumul_ddd_data.txt containing the energies to be simulated, as well as ...
 * \param dataPath the directory where the data file is stored
 * \return ...
 */
EnergyStruct energyReader(const std::string& dataPath);

#endif // ENERGY_READER_H
