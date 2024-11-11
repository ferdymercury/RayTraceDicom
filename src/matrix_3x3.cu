/**
 * \file
 * \brief Matrix3x3 class implementation
 */

#include "matrix_3x3.cuh"
#include "helper_math.h"
#include "helper_float3.cuh"

CUDA_CALLABLE_MEMBER Matrix3x3::Matrix3x3(const float3 a0, const float3 a1, const float3 a2) : r0(a0), r1(a1), r2(a2) {}

CUDA_CALLABLE_MEMBER Matrix3x3::Matrix3x3(const float s00, const float s01, const float s02, const float s10, const float s11, const float s12,
        const float s20, const float s21, const float s22) : r0(make_float3(s00, s01, s02)), r1(make_float3(s10, s11, s12)),
        r2(make_float3(s20, s21, s22)) {}

CUDA_CALLABLE_MEMBER Matrix3x3::Matrix3x3(const float s00, const float s11, const float s22) : r0(make_float3(s00, 0.0f, 0.0f)),
        r1(make_float3(0.0f, s11, 0.0f)), r2(make_float3(0.0f, 0.0f, s22)) {}

CUDA_CALLABLE_MEMBER Matrix3x3::Matrix3x3(const float3 a) : Matrix3x3(a.x,a.y,a.z) {}

CUDA_CALLABLE_MEMBER Matrix3x3::Matrix3x3(const float s) : Matrix3x3(s,s,s) {}

CUDA_CALLABLE_MEMBER Matrix3x3::Matrix3x3(float* const ptr) : Matrix3x3(ptr[0], ptr[1], ptr[2],ptr[3], ptr[4], ptr[5],ptr[6], ptr[7], ptr[8]) {}

CUDA_CALLABLE_MEMBER float3 Matrix3x3::operator*(const float3 a) const {
    Matrix3x3 m = *this;
    return make_float3(dot(m.row0(),a), dot(m.row1(),a), dot(m.row2(),a));
}

CUDA_CALLABLE_MEMBER Matrix3x3 Matrix3x3::operator*(const Matrix3x3 m) const {
    Matrix3x3 m1 = *this;
    Matrix3x3 m2 = m.transpose();
    return Matrix3x3(dot(m1.row0(), m2.row0()), dot(m1.row0(), m2.row1()), dot(m1.row0(), m2.row2()),
                    dot(m1.row1(), m2.row0()), dot(m1.row1(), m2.row1()), dot(m1.row1(), m2.row2()),
                    dot(m1.row2(), m2.row0()), dot(m1.row2(), m2.row1()), dot(m1.row2(), m2.row2()));
}

CUDA_CALLABLE_MEMBER void Matrix3x3::print() const {
    print_float3(r0);
    print_float3(r1);
    print_float3(r2);
}

CUDA_CALLABLE_MEMBER float Matrix3x3::det() const {
    return (r0.x*(r1.y*r2.z-r1.z*r2.y) - r0.y*(r1.x*r2.z-r1.z*r2.x) + r0.z*(r1.x*r2.y-r1.y*r2.x));
}

CUDA_CALLABLE_MEMBER Matrix3x3 Matrix3x3::inverse() const {
    float oneOverDet = 1./this->det();
    return Matrix3x3(r1.y*r2.z-r1.z*r2.y, r0.z*r2.y-r0.y*r2.z, r0.y*r1.z-r0.z*r1.y,
                    r1.z*r2.x-r1.x*r2.z, r0.x*r2.z-r0.z*r2.x, r0.z*r1.x-r0.x*r1.z,
                    r1.x*r2.y-r1.y*r2.x, r0.y*r2.x-r0.x*r2.y, r0.x*r1.y-r0.y*r1.x)*oneOverDet;
}

CUDA_CALLABLE_MEMBER Matrix3x3 Matrix3x3::transpose() const {
    return Matrix3x3(r0.x, r1.x, r2.x, r0.y, r1.y, r2.y, r0.z, r1.z, r2.z);
}

CUDA_CALLABLE_MEMBER float3 Matrix3x3::row0() const {return r0;}
CUDA_CALLABLE_MEMBER float3 Matrix3x3::row1() const {return r1;}
CUDA_CALLABLE_MEMBER float3 Matrix3x3::row2() const {return r2;}
