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
        __m256 load_k(const float* const, size_t);                      // masked load for remainder calculations
        void store_k(float*, __m256, size_t);                           // masked store for remainder calcs

        float _mm256_sum_ps(__m256 a);                                  // intermediate sum function
    }

    void setzero(Matrix&);                                              // used for accumulations and matrix multiplication
    
    float sum(const Matrix&);
    float dot(const Matrix&, const Matrix&);

    void add(const Matrix&, const Matrix&, Matrix&);
    void sub(const Matrix&, const Matrix&, Matrix&);                    // used for error calculation
    void mul(float, const Matrix&, Matrix&);
    void mul(const Matrix&, const Matrix&, Matrix&);
    void comp_div(Matrix&, float);                                      // used for averaging

    float square_sum(const Matrix&);                                    // used for mse calculation
    void mult_sum();

    void sigmoid(Matrix&);                                              // used for binary logistic regression
    void softmax(Matrix&);                                              // used for multi logistic regression

    void mult_comp_add(Matrix&, float, const Matrix&);
    void mult_comp_sub(Matrix&, float, const Matrix&);                  // used for gradient descent

    void cross_mul_bycol(const Matrix&, const Matrix&, Matrix&);        // used for prediction when #features >> #classes => #rows >> #cols
    void cross_mul_byrow(const Matrix&, const Matrix&, Matrix&);        // used for prediction when #features << #classes => #rows << #cols
    void cross_mul_to_N(const Matrix&, const Matrix&, Matrix&);         // used for prediction on regression models
    void cross_mul_to_M(const Matrix&, const Matrix&, Matrix&);         // used for gradient since error vector is 1d
}

#endif