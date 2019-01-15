/**
 * \file
 * \brief Matrix3x3 class declaration
 */
#ifndef MATRIX_3x3_CUH
#define MATRIX_3x3_CUH

#include "cuda_member.cuh"
#include "vector_types.h"

/**
 * \brief 3x3 matrix of floats
 */
struct Matrix3x3 {
    float3 r0;///< Row 0
    float3 r1;///< Row 1
    float3 r2;///< Row 2

    /**
     * \brief Struct constructor based on matrix rows
     * \param a0 ...
     * \param a1 ...
     * \param a2 ...
     */
    CUDA_CALLABLE_MEMBER Matrix3x3(const float3 a0, const float3 a1, const float3 a2);

    /**
     * \brief Struct constructor of a full matrix
     * \param s00 ...
     * \param s01 ...
     * \param s02 ...
     * \param s10 ...
     * \param s11 ...
     * \param s12 ...
     * \param s20 ...
     * \param s21 ...
     * \param s22 ...
     */
    CUDA_CALLABLE_MEMBER Matrix3x3(const float s00, float s01, const float s02, const float s10, const float s11, const float s12, const float s20, const float s21, const float s22);

    /**
     * \brief Struct constructor of a diagonal matrix
     * \param s00 ...
     * \param s11 ...
     * \param s22 ...
     */
    CUDA_CALLABLE_MEMBER Matrix3x3(const float s00, const float s11, const float s22);

    /**
     * \brief Struct constructor of a diagonal matrix
     * \param a ...
     */
    CUDA_CALLABLE_MEMBER Matrix3x3(const float3 a);

    /**
     * \brief Struct constructor of a diagonal matrix with same elements
     * \param s ...
     */
    CUDA_CALLABLE_MEMBER Matrix3x3(const float s);

    /**
     * \brief Struct constructor of a full matrix
     * \param ptr ...
     */
    CUDA_CALLABLE_MEMBER Matrix3x3(float* const ptr);

    /**
     * \brief Product operator of 3x3 matrix * float3
     * \param a ...
     * \return a float3
     */
    CUDA_CALLABLE_MEMBER float3 operator*(const float3 a) const;

    /**
     * \brief Product operator of 3x3 matrix * 3x3 matrix
     * \param m ...
     * \return a Matrix3x3
     */
    CUDA_CALLABLE_MEMBER Matrix3x3 operator*(const Matrix3x3 m) const;

    /**
     * \brief Calculates the determinant of the 3x3 matrix
     * \return the determinant as float3
     */
    CUDA_CALLABLE_MEMBER float det() const;

    /**
     * \brief Calculates the inverse of the 3x3 matrix
     * \return the inverse as Matrix3x3
     * \todo Check if inverse exists, i.e. det!=0
     */
    CUDA_CALLABLE_MEMBER Matrix3x3 inverse() const;

    /**
     * \brief Calculates the transpose of the 3x3 matrix
     * \return the transpose as Matrix3x3
     */
    CUDA_CALLABLE_MEMBER Matrix3x3 transpose() const;

    /**
     * \brief Prints the 3x3 matrix
     */
    CUDA_CALLABLE_MEMBER void print() const;

};

#endif // MATRIX_3x3_CUH
