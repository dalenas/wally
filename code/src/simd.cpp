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
    // for more features, less classes: dataset is row-major, weights are column-major
void matrix_mul(Matrix& Y, const Matrix& A, const Matrix& B) {
    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();

    const size_t N = A.rows();
    const size_t M = A.cols();
    const size_t K = B.cols();

    size_t r = M % 8;                        // Remaining feature count that can't create a full __m256 object
    for(size_t i = 0; i < N; ++i) {          // Per point
        const float* a_row = a + i*M;

        for(size_t k = 0; k < K; ++k) {      // Per class
            const float* b_col = b + k*M;

            size_t j = 0;
            __m256 y_vec = _mm256_setzero_ps();
            for(; j + 8 <= M; j += 8) {      // 8 features at a time
                __m256 a_vec = _mm256_loadu_ps(a_row + j);     // load A(i, j) - A(i, j+7), or the ith datapoint's j-j+7 features
                __m256 b_vec = _mm256_loadu_ps(b_col + j);     // load B(j, k) - B(j+7, k), or the kth class's j-j+7 weights

                 y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);
            }

            if(r != 0) {
                __m256 a_vec = SIMD_::load_k(a_row + j, r);
                __m256 b_vec = SIMD_::load_k(b_col + j, r);

                y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);
            }

            y[i*K + k] = SIMD_::_mm256_sum_ps(y_vec);
        }
    }
}

// mul2 is temporary name
    // Predict per feature
    // weights matrix needs to be row-major
    // so for less features, more classes: dataset and weight matrices are both row major
void matrix_mul2(Matrix& Y, const Matrix& A, const Matrix& B) {
    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();

    const size_t N = A.rows();
    const size_t M = A.cols();
    const size_t K = B.cols();

    size_t r = B.cols() % 8;
    for(size_t i = 0; i < N; ++i) {          // per point
        const float* a_row = a + i*M;
        float* y_row = y + i*K;

        for(size_t j = 0; j < M; ++j) {      // per feature
            const float* b_row = b + j*K;

            size_t k = 0;
            __m256 a_vec = _mm256_set1_ps(*(a_row + j));
            for(; k + 8 <= K; k += 8) {      // 8 classes at a time
                __m256 b_vec = _mm256_loadu_ps(b_row + k);
                __m256 y_vec = _mm256_loadu_ps(y_row + k);
                
                y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);

                _mm256_storeu_ps(y_row + k, y_vec);
            }

            if(r != 0) {
                __m256 b_vec = SIMD_::load_k(b_row + k, r);
                __m256 y_vec = SIMD_::load_k(y_row + k, r);

                y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);

                SIMD_::store_k(y_row + k, y_vec);         // Write store_k for this
            }
        }
    }
}

void mult_comp_sub(Matrix&, float, const Matrix&) {

}