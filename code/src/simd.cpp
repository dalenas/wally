#include "../lib/simd.h"

using namespace SIMD;

__m256 SIMD_::load(const float* const a) {
    return _mm256_loadu_ps(a);
}

__m256 SIMD_::load_k(const float* const a, size_t count) {
    __m256i mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(masks[count]));
    return _mm256_maskload_ps(a, mask);
}

void SIMD_::store(Matrix& X, __m256 X_256) {

}

float SIMD_::_mm256_sum_ps(__m256 a) {
    __m128 high = _mm256_castps256_ps128(a);
    __m128 low = _mm256_extractf32x4_ps(a, 1);

    __m128 sum_128 = _mm_add_ps(high, low);

    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);

    float sum = _mm_cvtss_f32(sum_128);
}

float SIMD_::_mm256_fmsum_ps(__m256 a, __m256 b) {
    
}

float sum(const Matrix& X) {
    const float* const x = X.data();

    size_t i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    for(; i + 8 <= X.size(); i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);

        sum_vec = _mm256_add_ps(sum_vec, x_vec);
    }

    float sum = _mm256_sum_ps(sum_vec);
    for(size_t r = i; r < X.size(); ++r)
        sum += x[r];

    return sum;
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

void mult_comp_add(Matrix& Y, float k, const Matrix& A) {
    const float* const a = A.data();
    float* const y = Y.data();

    __m256 k_vec = _mm256_set1_ps(k);

    size_t i = 0;
    for(; i + 8 <= A.size(); i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);

        y_vec = _mm256_fmadd_ps(k_vec, a_vec, y_vec);

        _mm256_storeu_ps(y+i, y_vec);
    }

    for(size_t r = i; r < A.size(); ++r)
        y[r] += k * a[r];
}

// A is row major, B is column major
    // Predict per class
void matrix_mul(Matrix& Y, const Matrix& A, const Matrix& B) {
    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();

    size_t r = A.cols() % 8;                        // Remaining feature count that can't create a full __m256 object
    for(size_t i = 0; i < A.rows(); ++i) {          // Per point
        for(size_t k = 0; k < B.cols(); ++k) {      // Per class
            size_t j = 0;
            for(; j + 8 <= A.cols(); j += 8) {      // 8 features at a time
                __m256 a_vec = _mm256_loadu_ps(a + i*A.cols() + j);     // load A(i, j) - A(i, j+7), or the ith datapoint's j-j+7 features
                __m256 b_vec = _mm256_loadu_ps(b + k*B.rows() + j);     // load B(j, k) - B(j+7, k), or the kth class's j-j+7 weights

                y[i*B.cols() + k] += SIMD_::_mm256_fmsum_ps(a_vec, b_vec);
            }

            __m256 a_tail = SIMD_::load_k(a + i*A.cols() + j, r);
            __m256 b_tail = SIMD_::load_k(b + k*B.rows() + j, r);

            y[i*B.cols() + k] += SIMD_::_mm256_fmsum_ps(a_tail, b_tail);
        }
    }
}

// mul2 is temporary name
    // Predict per feature
void matrix_mul2(Matrix& Y, const Matrix& A, const Matrix& B) {
    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();
}

void mult_comp_sub(Matrix&, float, const Matrix&) {

}