#include "../lib/simd.h"

__m256 SIMD::SIMD_::load_k(const float* const a, size_t k) {
    return _mm256_maskload_ps(a, masks[k]);
}

void SIMD::SIMD_::store_k(float* a, __m256 b, size_t k) {
    _mm256_maskstore_ps(a, masks[k], b);
}

float SIMD::SIMD_::_mm256_sum_ps(__m256 a) {
    __m128 high = _mm256_castps256_ps128(a);
    __m128 low = _mm256_extractf32x4_ps(a, 1);

    __m128 sum_128 = _mm_add_ps(high, low);

    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);

    return _mm_cvtss_f32(sum_128);
}

float SIMD::sum(const Matrix& X) {
    const float* const x = X.data();

    size_t i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    for(; i + 8 <= X.size(); i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);

        sum_vec = _mm256_add_ps(sum_vec, x_vec);
    }

    float sum = SIMD_::_mm256_sum_ps(sum_vec);
    for(size_t r = i; r < X.size(); ++r)
        sum += x[r];

    return sum;
}

float SIMD::dot(const Matrix&) {

}

void SIMD::add(const Matrix& A, const Matrix& B, Matrix& Y) {
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

void SIMD::sub(const Matrix& A, const Matrix& B, Matrix& Y) {
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

void SIMD::mul(float k, const Matrix& A, Matrix& Y) {
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

void SIMD::mul(const Matrix& A, const Matrix& B, Matrix& Y) {
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

void SIMD::mult_comp_add(Matrix& Y, float k, const Matrix& A) {
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

void SIMD::zero(Matrix& Y) {
    float* const y = Y.data();
    
    const size_t N = Y.size();
    const size_t remainder = N % 8;

    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for(; i + 8 <= N; i += 8)
        _mm256_storeu_ps(y + i, zero);
    if(remainder != 0)
        SIMD_::store_k(y + i, zero, remainder);
}

void SIMD::cross_mul_byrow(const Matrix& A, const Matrix& B, Matrix& Y) {
    zero(Y);

    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();

    const size_t N = A.rows();
    const size_t M = A.cols();
    const size_t K = B.cols();

    size_t remainder = M % 8;
    for(size_t i = 0; i < N; ++i) {
        const float* const a_row = a + i*M;

        for(size_t k = 0; k < K; ++k) {
            const float* const b_col = b + k*M;

            size_t j = 0;
            __m256 y_vec = _mm256_setzero_ps();
            for(; j + 8 <= M; j += 8) {
                __m256 a_vec = _mm256_loadu_ps(a_row + j);
                __m256 b_vec = _mm256_loadu_ps(b_col + j);

                 y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);
            }

            if(remainder != 0) {
                __m256 a_vec = SIMD_::load_k(a_row + j, remainder);
                __m256 b_vec = SIMD_::load_k(b_col + j, remainder);

                y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);
            }

            y[i*K + k] = SIMD_::_mm256_sum_ps(y_vec);
        }
    }
}

void SIMD::cross_mul_bycol(const Matrix& A, const Matrix& B, Matrix& Y) {
    zero(Y);

    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();

    const size_t N = A.rows();
    const size_t M = A.cols();
    const size_t K = B.cols();

    size_t remainder = B.cols() % 8;
    for(size_t i = 0; i < N; ++i) {
        const float* const a_row = a + i*M;
        float* y_row = y + i*K;

        for(size_t j = 0; j < M; ++j) {
            const float* const b_row = b + j*K;

            size_t k = 0;
            __m256 a_vec = _mm256_set1_ps(*(a_row + j));
            for(; k + 8 <= K; k += 8) {
                __m256 b_vec = _mm256_loadu_ps(b_row + k);
                __m256 y_vec = _mm256_loadu_ps(y_row + k);
                
                y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);

                _mm256_storeu_ps(y_row + k, y_vec);
            }

            if(remainder != 0) {
                __m256 b_vec = SIMD_::load_k(b_row + k, remainder);
                __m256 y_vec = SIMD_::load_k(y_row + k, remainder);

                y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);

                SIMD_::store_k(y_row + k, y_vec, remainder);
            }
        }
    }
}

void SIMD::cross_mul_to_N(const Matrix& A, const Matrix& B, Matrix& Y) {
    zero(Y);

    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();

    const size_t N = A.rows();
    const size_t M = A.cols();

    size_t remainder = M % 8;
    for(size_t i = 0; i < N; ++i) {
        const float* const a_row = a + i*M;

        size_t j = 0;
        __m256 y_vec = _mm256_setzero_ps();
        for(; j + 8 <= M; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(a_row + j);
            __m256 b_vec = _mm256_loadu_ps(b + j);
            
            y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);
        }

        if(remainder != 0) {
            __m256 a_vec = SIMD_::load_k(a_row + j, remainder);
            __m256 b_vec = SIMD_::load_k(b + j, remainder);

            y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);
        }

        y[i] = SIMD_::_mm256_sum_ps(y_vec);
    }
}

void SIMD::cross_mul_to_M(const Matrix& A, const Matrix& B, Matrix& Y) {
    zero(Y);

    const float* const a = A.data();
    const float* const b = B.data();
    float* const y = Y.data();

    const size_t N = A.rows();
    const size_t M = A.cols();

    size_t remainder = M % 8;
    for(size_t i = 0; i < N; ++i) {
        const float* const a_row = a + i*M;

        size_t j = 0;
        __m256 b_vec = _mm256_set1_ps(*(b + i));
        for(; j + 8 <= M; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(a_row + j);
            __m256 y_vec = _mm256_loadu_ps(y + j);

            y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);

            _mm256_storeu_ps(y + j, y_vec);
        }

        if(remainder != 0) {
            __m256 a_vec = SIMD_::load_k(a_row + j, remainder);
            __m256 y_vec = SIMD_::load_k(y + j, remainder);

            y_vec = _mm256_fmadd_ps(a_vec, b_vec, y_vec);

            SIMD_::store_k(y + j, y_vec, remainder);
        }
    }
}

void SIMD::mult_comp_sub(Matrix&, float, const Matrix&) {

}