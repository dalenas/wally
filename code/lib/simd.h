#ifndef SIMD_H
#define SIMD_H

#include <immintrin.h>

#include "matrix.h"

alignas(32) constexpr int masks[9][8] = {
    { 0,  0,  0,  0,  0,  0,  0,  0}, 
    {-1,  0,  0,  0,  0,  0,  0,  0}, 
    {-1, -1,  0,  0,  0,  0,  0,  0}, 
    {-1, -1, -1,  0,  0,  0,  0,  0}, 
    {-1, -1, -1, -1,  0,  0,  0,  0}, 
    {-1, -1, -1, -1, -1,  0,  0,  0}, 
    {-1, -1, -1, -1, -1, -1,  0,  0}, 
    {-1, -1, -1, -1, -1, -1, -1,  0}, 
    {-1, -1, -1, -1, -1, -1, -1, -1}
};

namespace SIMD {
    namespace SIMD_ {
        __m256 load_k(const float* const, size_t);

        float _mm256_sum_ps(__m256 a);
        float _mm256_fmsum_ps(__m256 a, __m256 b);
    }
    
    float sum(const Matrix&);
    float dot(const Matrix&);

    void add(const Matrix&, const Matrix&, Matrix&);
    void sub(const Matrix&, const Matrix&, Matrix&);
    void mult(float, const Matrix&, Matrix&);
    void mult(const Matrix&, const Matrix&, Matrix&);

    void mult_sum();

    void mult_comp_add(Matrix&, float, const Matrix&);
    void matrix_mul(Matrix&, const Matrix&, const Matrix&);
    void matrix_mul2(Matrix&, const Matrix&, const Matrix&);
    void mult_comp_sub(Matrix&, float, const Matrix&);
}

#endif