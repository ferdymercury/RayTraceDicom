/**
 * \file
 * \brief HostPinnedImage3D class declaration
 */
#ifndef HOST_IMAGE_3D_CUH
#define HOST_IMAGE_3D_CUH

#include "float3_affine_transform.cuh"

/**
 * \brief Templated class containing a 3D matrix of data type T
 */
template <typename T>
class HostPinnedImage3D {
public:
    /**
     * \brief Class constructor that stores an input 3D matrix via cudaHostRegister
     * \param imagePtr a pointer to the 3D matrix of data
     * \param dimensions a triad of unsigned integers, namely the number of bins along axis x, y and z
     */
    HostPinnedImage3D(T* const imagePtr, const uint3 dimensions) : imPtr(imagePtr), dims(dimensions) {
        cudaHostRegister((void*) imPtr, dims.x*dims.y*dims.z*sizeof(T), 0);
    }

    /**
     * \brief Class destructor, unregistering the 3D matrix
     */
    ~HostPinnedImage3D() {
        cudaHostUnregister((void*) imPtr);
    }

    /**
     * \brief Get a pointer to the 3D matrix
     * \return A pointer
     */
    T* getImData() const {return imPtr;}

    /**
     * \brief Get the matrix dimensions
     * \return triad of unsigned integers, namely the number of bins along axis x, y and z
     */
    uint3 getDims() const {return dims;}

private:
    T* const imPtr;     ///< Pointer to the 3D matrix (memory not owned by the class)
    const uint3 dims;   ///< Matrix dimensions (bins along x y z)
};

/**
 * \brief Template class containing a 3D matrix oriented in space (affine transform)
 */
template <typename T>
class HostPinnedOrientedImage3D : public HostPinnedImage3D<T> {
public:
    /**
     * \brief Class constructor that stores an input 3D matrix and its associated affine transform
     * \param imagePtr a pointer to the 3D matrix of data
     * \param dimensions a triad of unsigned integers, namely the number of bins along axis x, y and z
     * \param imageIdxToWorld an affine transform
     */
    HostPinnedOrientedImage3D(T* const imagePtr, const uint3 dimensions, const Float3AffineTransform imageIdxToWorld) :
    HostPinnedImage3D<T>(imagePtr, dimensions), iITW(imageIdxToWorld) {}

    /**
     * \brief Class constructor that stores an HostPinnedImage3D and its associated affine transform
     * \param hostIm3D the host pinned image base class
     * \param imageIdxToWorld an affine transform
     */
    HostPinnedOrientedImage3D(const HostPinnedImage3D<T> hostIm3D, const Float3AffineTransform imageIdxToWorld) :
    HostPinnedImage3D<T>(hostIm3D), iITW(imageIdxToWorld) {}

    Float3AffineTransform getTransf() const {return iITW;}

private:
    const Float3AffineTransform iITW;
};

#endif // HOST_IMAGE_3D_CUH
