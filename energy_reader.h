/**
 * \file
 * \brief energyReader function declaration
 */
#ifndef ENERGY_READER_H
#define ENERGY_READER_H

struct EnergyStruct;
#include <string>

/**
 * \brief Reads several data files proton_cumul_ddd_data.txt, density_Schneider2000_adj.txt, HU_to_SP_H&N_adj.txt, radiation_length_inc_water.txt, containing the energies to be simulated, as well as the cumulative integral depth dose distributions for each energy, density /stopping power / radiation length in water as function of HU. Optionally also nuclear_weights_and_sigmas_*.txt depending on if nuclear corrections are enabled
 * \param dataPath the directory where the data file is stored
 * \return the file parsed as an EnergyStruct containing all the info
 */
EnergyStruct energyReader(const std::string& dataPath);

#endif // ENERGY_READER_H
