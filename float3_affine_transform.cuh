/**
 * \file
 * \brief Float3AffineTransform class declaration
 */
#ifndef FLOAT3_AFFINE_TRANSFORM_CUH
#define FLOAT3_AFFINE_TRANSFORM_CUH

#include "cuda_member.cuh"
#include "helper_float3.cuh"
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
     * \brief ...
     * \param point ...
     * \return transformed point as float3
     */
    CUDA_CALLABLE_MEMBER float3 transformPoint(const float3 point) const;

    /**
     * \brief ...
     * \param vector ...
     * \return transformed vector as float3
     */
    CUDA_CALLABLE_MEMBER float3 transformVector(const float3 vector) const;

    /**
     * \brief ...
     * \return an instance of the inverse affine transform
     */
    Float3AffineTransform inverse() const;

    /**
     * \brief ...
     * \param toIdx ...
     * \return void
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
     * \return void
     */
    void print() const;

    /**
     * \brief Concatenate two affine transforms
     * \param t1 ...
     * \param t2 ...
     * \return Resulting affine transform instance
     */
    friend Float3AffineTransform concatFloat3AffineTransform(const Float3AffineTransform t1, const Float3AffineTransform t2);
};
#endif // FLOAT3_AFFINE_TRANSFORM_CUH

//Float3AffineTransform::Float3AffineTransform()
//Float3AffineTransform::Float3AffineTransform(Matrix3x3 mIn, float3 vIn)
//float3 Float3AffineTransform::transformPoint(float3 point)
//float3 Float3AffineTransform::transformVector(float3 vector)
//Float3AffineTransform Float3AffineTransform::inverse()
//Matrix3x3 Float3AffineTransform::getMatrix()
//float3 Float3AffineTransform::getOffset()
//void Float3AffineTransform::print()
//Float3AffineTransform concatFloat3AffineTransform(Float3AffineTransform t1, Float3AffineTransform t2)
