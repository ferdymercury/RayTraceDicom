/**
 * \file
 * \brief Float3AffineTransform class declaration
 */
#ifndef FLOAT3_AFFINE_TRANSFORM_CUH
#define FLOAT3_AFFINE_TRANSFORM_CUH

#include "cuda_member.cuh"
#include "matrix_3x3.cuh"

/**
 * \brief Affine transform class defined by a 3x3 matrix and a float3
 */
class Float3AffineTransform {
private:
    const Matrix3x3 m;  ///< 3x3 matrix containing orientation
    float3 v;           ///< 3d vector containing offset position

public:

    /**
     * \brief Default constructor. Sets m to identity and v to zero.
     */
    Float3AffineTransform();

    /**
     * \brief Constructor of affine transform
     * \param mIn the orientation matrix
     * \param vIn the offset vector
     */
    Float3AffineTransform(const Matrix3x3 mIn, const float3 vIn);

    /**
     * \brief Copy constructor of affine transform
     * \param in the transform to copy
     */
    Float3AffineTransform(const Float3AffineTransform& in);

    /**
     * \brief Transform a point according to internal affine matrix
     * \param point the 3D point to transform
     * \return transformed point as float3
     */
    CUDA_CALLABLE_MEMBER float3 transformPoint(const float3 point) const;

    /**
     * \brief Transform a vector according to internal affine matrix
     * \param vector the 3D vector to transform
     * \return transformed vector as float3
     */
    CUDA_CALLABLE_MEMBER float3 transformVector(const float3 vector) const;

    /**
     * \brief Calculate the inverse of current transform
     * \return an instance of the inverse affine transform
     */
    Float3AffineTransform inverse() const;

    /**
     * \brief Activate the index from start-at-one counting convention to start-at-zero
     * \param toIdx true to change from 1 to 0, false to do the opposite
     */
    void oneBasedToZeroBased(const bool toIdx);

    /**
     * \brief gets the stored orientation matrix
     * \return a Matrix3x3
     */
    CUDA_CALLABLE_MEMBER Matrix3x3 getMatrix() const;

    /**
     * \brief gets the stored offset
     * \return a float3
     */
    CUDA_CALLABLE_MEMBER float3 getOffset() const;

    /**
     * \brief Print affine transformation to console
     */
    void print() const;

    /**
     * \brief Concatenate two affine transforms
     * \param t1 first transform
     * \param t2 second transform
     * \return Resulting affine transform instance
     */
    friend Float3AffineTransform concatFloat3AffineTransform(const Float3AffineTransform t1, const Float3AffineTransform t2);
};
#endif // FLOAT3_AFFINE_TRANSFORM_CUH
