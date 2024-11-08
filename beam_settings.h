/**
 * \file
 * \brief BeamSettings class declaration
 */
#ifndef BEAM_SETTINGS_H
#define BEAM_SETTINGS_H

#include "host_image_3d.cuh"
#include "float3_affine_transform.cuh"
#include "float3_idx_transform.cuh"
#include <vector>
//#include "float3_from_fan_transform.cuh"

/**
 * \brief Struct containing all beam settings
 */
class BeamSettings {
public:
    /**
     * @brief Constructor
     * @param spotWeights Non-owning pointer to pinned memory containing the spot weight maps for each energy layer. Energy layers are stacked along the slowest dimension, i.e. spotWeights[i,j,k] contains the weight of a spot with x_idx=i, y_idx=j and energy_layer_idx=k.
     * @param beamEnergies Reference to vector containing the beam energies for each energy layer.
     * @param spotSigmas Reference to vector containing the spot sigmas (x and y) at iso in air for each energy layer.
     * @param raySpacing Spacing in x and y between adjacent raytracing rays at iso (determines the lateral resolution of the dose calculated in gantry coordinates).
     * @param tracerSteps The number of steps to carry out the raytracing for (given by the distance between near instersect and far intersect between image and the beam, divided by the raytrace step size).
     * @param sourceDist The apparent source to iso distance for the beam along x and y, used to calculate the divergence of spots.
     * @param spotIdxToGantry Transform from spot index to gantry coordinates. Transforming [i, j, k] gives the position, in gantry coordinates, of spot with x_idx=i, y_idx=j and at ray trace step k (i.e. the delta and offset in last dimension give the raytracing step length and the position along gantry z at which to start the raytracing).
     * @param gantryToImIdx Reference to affine transformation from gantry coordinates to indices in the patient image.
     * @param gantryToDoseIdx Reference to affine transformation from gantry coordinates to indices in the dose matrix.
     */
    BeamSettings(HostPinnedImage3D<float>* const spotWeights, const std::vector<float>& beamEnergies, const std::vector<float2>& spotSigmas, const float2 raySpacing, const unsigned int tracerSteps, const float2 sourceDist, const Float3IdxTransform spotIdxToGantry, const Float3AffineTransform& gantryToImIdx, const Float3AffineTransform& gantryToDoseIdx);

    /**
     * @brief spot weight maps for each energy layer.
     * @return Raw  pointer to 3D host-image, the energy layers are stacked along the slowest dimension, i.e. spotWeights[i,j,k] contains the weight of a spot with x_idx=i, y_idx=j and energy_layer_idx=k.
     */
    HostPinnedImage3D<float>* getWeights();

    /**
     * @brief Get energy layers
     * @return vector with energy for each layer
     */
    std::vector<float>& getEnergies();

    /**
     * @brief Get spatial spread of spots
     * @return vector of pairs (sigmax, sigmay) at iso in air for each energy layer
     */
    std::vector<float2>& getSpotSigmas();

    /**
     * @brief Get spacing between adjacent raytracing rays at iso (determines the lateral resolution of the dose calculated in gantry coordinates).
     * @return spacing in x and y as a float2
     */
    float2 getRaySpacing() const;

    /**
     * @brief Get number of steps to carry out the raytracing for (given by the distance between near instersect and far intersect between image and the beam, divided by the raytrace step size).
     * @return number of steps
     */
    unsigned int getSteps() const;

    /**
     * @brief The apparent source to iso distance for the beam along x and y, used to calculate the divergence of spots.
     * @return source to iso distance in x and y, as float2
     */
    float2 getSourceDist() const;

    /**
     * @brief Get Transform from spot index to gantry coordinates.
     * Transforming [i, j, k] gives the position, in gantry coordinates, of spot with x_idx=i, y_idx=j and at ray trace step k (i.e. the delta and offset in last dimension give the raytracing step length and the position along gantry z at which to start the raytracing).
     * @return Transform matrix from spot index to gantry coordinates as Float3IdxTransform
     */
    Float3IdxTransform getSpotIdxToGantry() const;

    /**
     * @brief Get affine transformation matrix from gantry coordinates to indices in the patient image.
     * @return Float3AffineTransform
     */
    Float3AffineTransform getGantryToImIdx() const;

    /**
     * @brief Get affine transformation from gantry coordinates to indices in the dose matrix.
     * @return Float3AffineTransform
     */
    Float3AffineTransform getGantryToDoseIdx() const;

    //
    // brief ...
    // return ...
    //
    //Float3FromFanTransform getFITII() const;

    //
    // brief ...
    // return ...
    //
    //Float3FromFanTransform getFITDI() const;

private:
    HostPinnedImage3D<float>* const sWghts; ///< Non-owning pointer to spot weight map x,y for each energy layer
    std::vector<float> bEnergies;           ///< Energy for each layer
    std::vector<float2> sSigmas;            ///< (sigmax, sigmay) at iso in air for each energy layer
    float2 rSpacing;                        ///< spacing between adjacent raytracing rays at iso (determines the lateral resolution of the dose calculated in gantry coordinates).
    unsigned int steps;                     ///< Get number of steps to carry out the raytracing for (given by the distance between near instersect and far intersect between image and the beam, divided by the raytrace step size).
    float2 sDist;                           ///< apparent source to iso distance for the beam along x and y, used to calculate the divergence of spots.
    Float3IdxTransform sITG;                ///< Transform from spot index to gantry coordinates.Transforming [i, j, k] gives the position, in gantry coordinates, of spot with x_idx=i, y_idx=j and at ray trace step k (i.e. the delta and offset in last dimension give the raytracing step length and the position along gantry z at which to start the raytracing).
    Float3AffineTransform gTII;             ///< Affine transformation matrix from gantry coordinates to indices in the patient image.
    Float3AffineTransform gTDI;             ///< Affine transformation matrix from gantry coordinates to indices in the dose matrix.
    //Float3FromFanTransform fITII; ///< ...
    //Float3FromFanTransform fITDI; ///< ...
};

#endif // BEAM_SETTINGS_H
