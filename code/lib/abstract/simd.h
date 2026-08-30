#ifndef SIMD_H
#define SIMD_H

#include <immintrin.h>

#include "traits.h"

namespace Wally::Abstract::SIMD {
    namespace SIMD_ {
        alignas(32) inline const __m256i masks[9] = {
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

        __m256 load_k(const float* const, std::size_t);                      // masked load for remainder calculations
        __m256i load_k(const int* const, std::size_t);

        void store_k(float*, __m256, std::size_t);                           // masked store for remainder calcs
        void store_k(int*, __m256i, std::size_t);

        float _mm256_sum_ps(__m256);                                  // intermediate sum function
        int _mm256_sum_ps(__m256i);
    }

    template<typename Container>
    void setzero(Container&);

    template<typename Container>
    Container::element_type sum(const Container&);

    template<typename T>
    void sum(const Matrix<T>&, Vector<T>&);
    template<typename T>
    T dot(const Vector<T>&, const Vector<T>&);

    template<typename Container>
    void add(const Container::element_type, const Container&, Container&);
    template<typename Container>
    void sub(const Container::element_type, const Container&, Container&);
    template<typename Container>
    void mul(const Container::element_type, const Container&, Container&);
    template<typename Container>
    void div(const Container::element_type, const Container&, Container&);

    template<typename Container>
    void add(const Container&, const Container&, Container&);
    template<typename Container>
    void sub(const Container&, const Container&, Container&);
    template<typename Container>
    void mul(const Container&, const Container&, Container&);
    template<typename Container>
    void div(const Container&, const Container&, Container&);

     
















    /*
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
    */
}

#endif