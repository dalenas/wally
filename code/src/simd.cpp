#include "../lib/simd.h"

using namespace SIMD;

__m256 SIMD_::load(const Matrix& X) {

}

void SIMD_::store(Matrix& X, __m256 X_256) {

}

float sum(const Matrix& X) {
    
}

float dot(const Matrix&) {

}

void add(const Matrix& A, const Matrix& B, Matrix& Y) {
    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();

    size_t i = 0;
    for(; i + 8 <= A.size(); i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 y_vec = _mm256_add_ps(a_vec, b_vec);

        _mm256_storeu_ps(y + i, y_vec);
    }

    for(size_t r = i; r < A.size(); ++r)
        y[r] = a[r] + b[r];
}

void sub(const Matrix& A, const Matrix& B, Matrix& Y) {
    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();

    size_t i = 0;
    for(; i + 8 <= A.size(); i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 y_vec = _mm256_sub_ps(a_vec, b_vec);

        _mm256_storeu_ps(y + i, y_vec);
    }

    for(size_t r = i; r < A.size(); ++r)
        y[r] = a[r] - b[r];
}

void mult(float k, const Matrix& A, Matrix& Y) {
    const float* const a = A.data();
    float* const y = Y.data();

    __m256 k_vec = _mm256_set1_ps(k);

    size_t i = 0;
    for(; i + 8 <= A.size(); i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 y_vec = _mm256_mul_ps(k_vec, a_vec);

        _mm256_storeu_ps(y + i, y_vec);
    }

    for(size_t r = i; r < A.size(); ++r)
        y[r] = k * a[r];
}

void mult(const Matrix& A, const Matrix& B, Matrix& Y) {
    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();

    size_t i = 0;
    for(; i + 8 <= A.size(); i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 y_vec = _mm256_mul_ps(a_vec, b_vec);

        _mm256_storeu_ps(y + i, y_vec);
    }

    for(size_t r = i; r < A.size(); ++r)
        y[r] = a[r] * b[r];
}

void mult_comp_add(Matrix& Y, const Matrix& A, const Matrix& B) {
    
}

void mult_comp_sub(Matrix&, float, const Matrix&) {

}