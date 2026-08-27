#ifndef SIMD_H
#define SIMD_H

#include <immintrin.h>

#include "matrix.h"

alignas(32) const __m256i masks[9] = {
    _mm256_setr_epi32( 0,  0,  0,  0,  0,  0,  0,  0), 
    _mm256_setr_epi32(-1,  0,  0,  0,  0,  0,  0,  0), 
    _mm256_setr_epi32(-1, -1,  0,  0,  0,  0,  0,  0), 
    _mm256_setr_epi32(-1, -1, -1,  0,  0,  0,  0,  0), 
    _mm256_setr_epi32(-1, -1, -1, -1,  0,  0,  0,  0), 
    _mm256_setr_epi32(-1, -1, -1, -1, -1,  0,  0,  0), 
    _mm256_setr_epi32(-1, -1, -1, -1, -1, -1,  0,  0), 
    _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1,  0), 
    _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1)
};

namespace SIMD {
    namespace SIMD_ {
        __m256 load_k(const float* const, size_t);
        void store_k(float*, __m256, size_t);

        float _mm256_sum_ps(__m256 a);
    }

    void setzero(Matrix&);
    
    float sum(const Matrix&);
    float dot(const Matrix&);

    void add(const Matrix&, const Matrix&, Matrix&);
    void sub(const Matrix&, const Matrix&, Matrix&);                    // used for error calculation
    void mul(float, const Matrix&, Matrix&);
    void mul(const Matrix&, const Matrix&, Matrix&);
    void comp_div(Matrix&, float);                                      // used for averaging

    float square_sum(const Matrix&);                                    // used for mse calculation
    void mult_sum();

    // softmax -> calculatecalculate e^z, sum e^z per point, normalize each e^z for p
        // for e^z calculation:
            // case 1: calculate 8 zs first, then do mm256_exp for 8 e^z
            // case 2: calculate 8 e^xiwi at a time and use mm256_mul, then dot product to find e^z
    void softmax(Matrix&);

    void mult_comp_add(Matrix&, float, const Matrix&);
    void cross_mul_bycol(const Matrix&, const Matrix&, Matrix&);        // used for prediction when #features >> #classes => #rows >> #cols
    void cross_mul_byrow(const Matrix&, const Matrix&, Matrix&);        // used for prediction when #features << #classes => #rows << #cols
    void cross_mul_to_N(const Matrix&, const Matrix&, Matrix&);         // used for prediction on regression models
    void cross_mul_to_M(const Matrix&, const Matrix&, Matrix&);         // used for gradient since error vector is 1d
    void mult_comp_sub(Matrix&, float, const Matrix&);                  // used for gradient descent
}

#endif