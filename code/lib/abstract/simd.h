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

        __m256 _mm256_loadu(const float* const);
        __m256i _mm256_loadu(const int* const);

        __m256 _mm256_maskloadu(const float* const, std::size_t);                      // masked load for remainder calculations
        __m256i _mm256_maskloadu(const int* const, std::size_t);

        void _mm256_storeu(float*, __m256);
        void _mm256_storeu(int*, __m256i);

        void _mm256_maskstoreu(float*, __m256, std::size_t);                           // masked store for remainder calcs
        void _mm256_maskstoreu(int*, __m256i, std::size_t);

        __m256 _mm256_set1(float);
        __m256 _mm256_set1(int);

        __m256 _mm256_add(__m256, __m256);
        __m256 _mm256_add(__m256, __m256i);
        __m256 _mm256_add(__m256i, __m256);
        __m256i _mm256_add(__m256i, __m256i);

        __m256 _mm256_sub(__m256, __m256);
        __m256 _mm256_sub(__m256, __m256i);
        __m256 _mm256_sub(__m256i, __m256);
        __m256i _mm256_sub(__m256i, __m256i);

        __m256 _mm256_mul(__m256, __m256);
        __m256 _mm256_mul(__m256, __m256i);
        __m256 _mm256_mul(__m256i, __m256);
        __m256i _mm256_mul(__m256i, __m256i);

        __m256 _mm256_div(__m256, __m256);
        __m256 _mm256_div(__m256, __m256i);
        __m256 _mm256_div(__m256i, __m256);
        __m256 _mm256_div(__m256i, __m256i);
    }
    constexpr std::size_t WIDTH = 8;

    // Sets every element in the container to 0
    template<typename ContainerY>
    void setzero(ContainerY&);

    template<typename Container>
    float sum(const Container&);

    template<typename T>
    void sum(const Matrix<T>&, Vector<T>&);
    template<typename T>
    T dot(const Vector<T>&, const Vector<T>&);
    
    // Scalar broadcast operations
    template<typename S, typename ContainerA, typename ContainerY>
    void add(const S, const ContainerA&, ContainerY&);
    template<typename S, typename ContainerA, typename ContainerY>
    void sub(const S, const ContainerA&, ContainerY&);
    template<typename ContainerA, typename S, typename ContainerY>
    void sub(const ContainerA&, const S, ContainerY&);
    template<typename S, typename ContainerA, typename ContainerY>
    void mul(const S, const ContainerA&, ContainerY&);
    template<typename S, typename ContainerA, typename ContainerY>
    void div(const S, const ContainerA&, ContainerY&);
    template<typename ContainerA, typename S, typename ContainerY>
    void div(const ContainerA&, const S, ContainerY&);

    // Element-wise operations
    template<typename ContainerA, typename ContainerB, typename ContainerY>
    void add(const ContainerA&, const ContainerB&, ContainerY&);
    template<typename ContainerA, typename ContainerB, typename ContainerY>
    void sub(const ContainerA&, const ContainerB&, ContainerY&);
    template<typename ContainerA, typename ContainerB, typename ContainerY>
    void mul(const ContainerA&, const ContainerB&, ContainerY&);
    template<typename ContainerA, typename ContainerB, typename ContainerY>
    void div(const ContainerA&, const ContainerB&, ContainerY&);

    // Compound scalar operations
    template<typename ContainerY, typename S>
    void add(ContainerY&, const S);
    template<typename ContainerY, typename S>
    void sub(ContainerY&, const S);
    template<typename ContainerY, typename S>
    void mul(ContainerY&, const S);
    template<typename ContainerY, typename S>
    void div(ContainerY&, const S);


     
















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