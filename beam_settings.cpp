/**
 * \file
 * \brief BeamSettings class implementation
 */
#include "beam_settings.h"

BeamSettings::BeamSettings(const HostPinnedImage3D<float> spotWeights, const std::vector<float> beamEnergies, const std::vector<float2> spotSigmas, const float2 raySpacing, const unsigned int tracerSteps, const float2 sourceDist, const Float3IdxTransform spotIdxToGantry, const Float3AffineTransform gantryToImIdx, const Float3AffineTransform gantryToDoseIdx) :
sWghts(spotWeights), bEnergies(beamEnergies), sSigmas(spotSigmas), rSpacing(raySpacing), steps(tracerSteps), sDist(sourceDist), sITG(spotIdxToGantry), gTII(gantryToImIdx), gTDI(gantryToDoseIdx) {};

HostPinnedImage3D<float>& BeamSettings::getWeights() {return sWghts;}

std::vector<float>& BeamSettings::getEnergies() {return bEnergies;}

std::vector<float2>& BeamSettings::getSpotSigmas() {return sSigmas;}

float2 BeamSettings::getRaySpacing() const {return rSpacing;}

unsigned int BeamSettings::getSteps() const {return steps;}

float2 BeamSettings::getSourceDist() const {return sDist;}

Float3IdxTransform BeamSettings::getSpotIdxToGantry() const {return sITG;}

Float3AffineTransform BeamSettings::getGantryToImIdx() const {return gTII;}

Float3AffineTransform BeamSettings::getGantryToDoseIdx() const {return gTDI;}

//Float3FromFanTransform BeamSettings::getFITII() const {return fITII;}

//Float3FromFanTransform BeamSettings::getFITDI() const {return fITDI;}
