using namespace Wally;
using namespace Abstract;

__m256 SIMD::SIMD_::_mm256_loadu(const float* const addr) { return _mm256_loadu_ps(addr); }
__m256i SIMD::SIMD_::_mm256_loadu(const int* const addr) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(addr)); }

__m256 SIMD::SIMD_::_mm256_maskloadu(const float* const addr, std::size_t k) { return _mm256_maskload_ps(addr, masks[k]); }
__m256i SIMD::SIMD_::_mm256_maskloadu(const int* const addr, std::size_t k) { return _mm256_maskload_epi32(addr, masks[k]); }

void SIMD::SIMD_::_mm256_storeu(float* addr, __m256 a) { _mm256_storeu_ps(addr, a); }
void SIMD::SIMD_::_mm256_storeu(int* addr, __m256i a) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(addr), a); }

void SIMD::SIMD_::_mm256_maskstoreu(float* addr, __m256 a, std::size_t k) { _mm256_maskstore_ps(addr, masks[k], a); }
void SIMD::SIMD_::_mm256_maskstoreu(int* addr, __m256i a, std::size_t k) { _mm256_maskstore_epi32(addr, masks[k], a); }

__m256 SIMD::SIMD_::_mm256_set1(float k) { return _mm256_set1_ps(k); }
__m256i SIMD::SIMD_::_mm256_set1(int k) { return _mm256_set1_epi32(k); }            // may want to change to m256i, which would require if constexpr conditions in scalar operations

template<typename T>
auto SIMD::SIMD_::_mm256_setzero() {
    if constexpr(std::is_same_v<T, float>)
        return _mm256_setzero_ps();
    else if constexpr(std::is_same_v<T, int>)
        return _mm256_setzero_si256();
}

__m256 SIMD::SIMD_::_mm256_add(const __m256 a, const __m256 b) { return _mm256_add_ps(a, b); }
__m256 SIMD::SIMD_::_mm256_add(__m256 a, __m256i b) { 
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_add_ps(a, bf);
}
__m256 SIMD::SIMD_::_mm256_add(__m256i a, __m256 b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_add_ps(af, b);
}
__m256i SIMD::SIMD_::_mm256_add(__m256i a, __m256i b) { return _mm256_add_epi32(a, b); }

__m256 SIMD::SIMD_::_mm256_sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
__m256 SIMD::SIMD_::_mm256_sub(__m256 a, __m256i b) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_sub_ps(a, bf);
}
__m256 SIMD::SIMD_::_mm256_sub(__m256i a, __m256 b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_sub_ps(af, b);
}
__m256i SIMD::SIMD_::_mm256_sub(__m256i a, __m256i b) { return _mm256_sub_epi32(a, b); }

__m256 SIMD::SIMD_::_mm256_mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
__m256 SIMD::SIMD_::_mm256_mul(__m256 a, __m256i b) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_mul_ps(a, bf);
}
__m256 SIMD::SIMD_::_mm256_mul(__m256i a, __m256 b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_mul_ps(af, b);
}
__m256i SIMD::SIMD_::_mm256_mul(__m256i a, __m256i b) { return _mm256_mullo_epi32(a, b); }

__m256 SIMD::SIMD_::_mm256_div(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }
__m256 SIMD::SIMD_::_mm256_div(__m256 a, __m256i b) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_div_ps(a, bf);
}
__m256 SIMD::SIMD_::_mm256_div(__m256i a, __m256 b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_div_ps(af, b);
}
__m256 SIMD::SIMD_::_mm256_div(__m256i a, __m256i b) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_div_ps(af, bf);
}

float SIMD::SIMD_::_mm256_sum(__m256 a) {
    __m128 high = _mm256_castps256_ps128(a);
    __m128 low = _mm256_extractf128_ps(a, 1);

    __m128 sum_vec = _mm_add_ps(high, low);

    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);

    return _mm_cvtss_f32(sum_vec);
}

int SIMD::SIMD_::_mm256_sum(__m256i a) {
    __m128i high = _mm256_castsi256_si128(a);
    __m128i low = _mm256_extracti128_si256(a, 1);

    __m128i sum_vec = _mm_add_epi32(high, low);

    sum_vec = _mm_hadd_epi32(sum_vec, sum_vec);
    sum_vec = _mm_hadd_epi32(sum_vec, sum_vec);

    return _mm_cvtsi128_si32(sum_vec);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }

__m256 SIMD::SIMD_::_mm256_fmadd(__m256 a, __m256 b, __m256i c) {
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmadd_ps(a, b, cf);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256 a, __m256i b, __m256 c) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_fmadd_ps(a, bf, c);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256 a, __m256i b, __m256i c) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmadd_ps(a, bf, cf);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256i a, __m256 b, __m256 c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_fmadd_ps(af, b, c);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256i a, __m256 b, __m256i c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmadd_ps(af, b, cf);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256i a, __m256i b, __m256 c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_fmadd_ps(af, bf, c);
}

__m256 SIMD::SIMD_::_mm256_fmadd(__m256i a, __m256i b, __m256i c) { 
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmadd_ps(af, bf, cf);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256 a, __m256 b, __m256 c) { return _mm256_fmsub_ps(a, b, c); }

__m256 SIMD::SIMD_::_mm256_fmsub(__m256 a, __m256 b, __m256i c) {
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmsub_ps(a, b, cf);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256 a, __m256i b, __m256 c) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_fmsub_ps(a, bf, c);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256 a, __m256i b, __m256i c) {
    __m256 bf = _mm256_cvtepi32_ps(b);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmsub_ps(a, bf, cf);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256i a, __m256 b, __m256 c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_fmsub_ps(af, b, c);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256i a, __m256 b, __m256i c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmsub_ps(af, b, cf);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256i a, __m256i b, __m256 c) {
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    return _mm256_fmsub_ps(af, bf, c);
}

__m256 SIMD::SIMD_::_mm256_fmsub(__m256i a, __m256i b, __m256i c) { 
    __m256 af = _mm256_cvtepi32_ps(a);
    __m256 bf = _mm256_cvtepi32_ps(b);
    __m256 cf = _mm256_cvtepi32_ps(c);
    return _mm256_fmsub_ps(af, bf, cf);
}

__m256 SIMD::SIMD_::_mm256_abs(__m256 a) {
    const __m256 sign_vec = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(sign_vec, a);
}

__m256i SIMD::SIMD_::_mm256_abs(__m256i a) {
    return _mm256_abs_epi32(a);
}

__m256 SIMD::SIMD_::_mm256_rcp(__m256 a) {
    return _mm256_rcp_ps(a);
}

__m256 SIMD::SIMD_::_mm256_rcp(__m256i a) {
    __m256 af = _mm256_cvtepi32_ps(a);
    return _mm256_rcp_ps(af);
}

template<Container ContainerY>
void SIMD::setzero(ContainerY& Y) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const auto zero = SIMD_::_mm256_setzero<U>();
    for(; i < EDGE; i += 8)
        SIMD_::_mm256_storeu(y + i, zero);

    if(REMAINDER != 0)
        SIMD_::_mm256_maskstoreu(y + i, zero, REMAINDER);
}

template<typename T>
T SIMD::sum(const Vector<T>& A) {
    const T* const a = A.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    auto sum_vec = SIMD_::_mm256_setzero<T>();
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);

        sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);

        sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);
    }

    return SIMD_::_mm256_sum(sum_vec);
}

template<typename T>
void SIMD::sum(const Matrix<T>& A, Vector<T>& Y) {
    const T* const a = A.data();
    T* const y = Y.data();

    const std::size_t N = A.rows();
    const std::size_t M = A.cols();
    
    if(A.axis() == Major::row) {
        const std::size_t REMAINDER = M % WIDTH;
        const std::size_t EDGE = M - REMAINDER;

        for(std::size_t i = 0; i < N; ++i) {
            const T* const a_row = a + i*M;

            std::size_t j = 0;
            for(; j < EDGE; j += 8) {
                const auto a_vec = SIMD_::_mm256_loadu(a_row + j);
                auto sum_vec = SIMD_::_mm256_loadu(y + j);

                sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);

                SIMD_::_mm256_storeu(y + j, sum_vec);
            }

            if(REMAINDER != 0) {
                const auto a_vec = SIMD_::_mm256_maskloadu(a_row + j, REMAINDER);
                auto sum_vec = SIMD_::_mm256_maskloadu(y + j, REMAINDER);
                
                sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);

                SIMD_::_mm256_maskstoreu(y + j, sum_vec, REMAINDER);
            }
        }
    }
    else if(A.axis() == Major::col) {
        const std::size_t REMAINDER = N % WIDTH;
        const std::size_t EDGE = N - REMAINDER;

        for(std::size_t j = 0; j < M; ++j) {
            const T* const a_col = a + j*N;

            std::size_t i = 0;
            auto sum_vec = SIMD_::_mm256_setzero<T>();
            for(; i < EDGE; i += 8) {
                const auto a_vec = SIMD_::_mm256_loadu(a_col + i);
                
                sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);
            }

            if(REMAINDER != 0) {
                const auto a_vec = SIMD_::_mm256_maskloadu(a_col + i);

                sum_vec = SIMD_::_mm256_add(a_vec, sum_vec);
            }

            y[j] = SIMD_::_mm256_sum(sum_vec);
        }
    }
}

template<typename T>
T SIMD::dot(const Vector<T>& A, const Vector<T>& B) {
    const T* const a = A.data();
    const T* const b = B.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - WIDTH;

    std::size_t i = 0;
    auto dot_vec = SIMD_::_mm256_setzero<T>();
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto b_vec = SIMD_::_mm256_loadu(b + i);

        dot_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, dot_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);

        dot_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, dot_vec);
    }

    return SIMD_::_mm256_sum(dot_vec);
}

template<typename S, Container ContainerA, Container ContainerY>
void SIMD::add(const S k, const ContainerA& A, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const auto k_vec = SIMD_::_mm256_set1(k);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto y_vec = SIMD_::_mm256_add(k_vec, a_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto y_vec = SIMD_::_mm256_add(k_vec, a_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename S, Container ContainerA, Container ContainerY>
void SIMD::sub(const S k, const ContainerA& A, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const auto k_vec = SIMD_::_mm256_set1(k);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto y_vec = SIMD_::_mm256_sub(k_vec, a_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto y_vec = SIMD_::_mm256_sub(k_vec, a_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<Container ContainerA, typename S, Container ContainerY>
void SIMD::sub(const ContainerA& A, const S k, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const auto k_vec = SIMD_::_mm256_set1(k);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto y_vec = SIMD_::_mm256_sub(a_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto y_vec = SIMD_::_mm256_sub(a_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename S, Container ContainerA, Container ContainerY>
void SIMD::mul(const S k, const ContainerA& A, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const auto k_vec = SIMD_::_mm256_set1(k);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto y_vec = SIMD_::_mm256_mul(k_vec, a_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto y_vec = SIMD_::_mm256_mul(k_vec, a_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<typename S, Container ContainerA, Container ContainerY>
void SIMD::div(const S k, const ContainerA& A, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const float k_ps = static_cast<float>(k);
    const __m256 k_vec = SIMD_::_mm256_set1(k_ps);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const __m256 y_vec = SIMD_::_mm256_div(k_vec, a_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const __m256 y_vec = SIMD_::_mm256_div(k_vec, a_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<Container ContainerA, typename S, Container ContainerY>
void SIMD::div(const ContainerA& A, const S k, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const float k_ps = static_cast<float>(k);
    const __m256 k_vec = SIMD_::_mm256_set1(k_ps);
    std::size_t i = 0;
    for(; i < EDGE; i += 8) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const __m256 y_vec = SIMD_::_mm256_div(a_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const __m256 y_vec = SIMD_::_mm256_div(a_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<Container ContainerA, Container ContainerB, Container ContainerY>
void SIMD::add(const ContainerA& A, const ContainerB& B, ContainerY& Y) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using Y_traits = container_traits<ContainerY>;

    using S = typename A_traits::element_type;
    using T = typename B_traits::element_type;
    using U = typename Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type){
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto y_vec = SIMD_::_mm256_add(a_vec, b_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto y_vec = SIMD_::_mm256_add(a_vec, b_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        }
    } // need to add vector + matrix, matrix + vector
}

template<Container ContainerA, Container ContainerB, Container ContainerY>
void SIMD::sub(const ContainerA& A, const ContainerB& B, ContainerY& Y) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using Y_traits = container_traits<ContainerY>;

    using S = typename A_traits::element_type;
    using T = typename B_traits::element_type;
    using U = typename Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type) {
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto y_vec = SIMD_::_mm256_sub(a_vec, b_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto y_vec = SIMD_::_mm256_sub(a_vec, b_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        }
    } // need to add vector - matrix, matrix - vector
}

template<Container ContainerA, Container ContainerB, Container ContainerY>
void SIMD::mul(const ContainerA& A, const ContainerB& B, ContainerY& Y) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using Y_traits = container_traits<ContainerY>;

    using S = typename A_traits::element_type;
    using T = typename B_traits::element_type;
    using U = typename Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type) {
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto y_vec = SIMD_::_mm256_mul(a_vec, b_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto y_vec = SIMD_::_mm256_mul(a_vec, b_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        } // need to add vector * matrix, matrix * vector
    }
}

template<Container ContainerA, Container ContainerB, Container ContainerY>
void SIMD::div(const ContainerA& A, const ContainerB& B, ContainerY& Y) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using Y_traits = container_traits<ContainerY>;

    using S = typename A_traits::element_type;
    using T = typename B_traits::element_type;
    using U = typename Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type) {
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto y_vec = SIMD_::_mm256_div(a_vec, b_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto y_vec = SIMD_::_mm256_div(a_vec, b_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        }
    } // need to add matrix / vector, vector / matrix 
}

template<Container ContainerY, typename S>
void SIMD::add(ContainerY& Y, const S k) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const auto k_vec = SIMD_::_mm256_set1(k);
    for(; i < EDGE; i += 8) {
        auto y_vec = SIMD_::_mm256_loadu(y + i);
        y_vec = SIMD_::_mm256_add(y_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        auto y_vec = SIMD_::_mm256_maskloadu(y + i, REMAINDER);
        y_vec = SIMD_::_mm256_add(y_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<Container ContainerY, typename S>
void SIMD::sub(ContainerY& Y, const S k) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const auto k_vec = SIMD_::_mm256_set1(k);
    for(; i < EDGE; i += 8) {
        auto y_vec = SIMD_::_mm256_loadu(y + i);
        y_vec = SIMD_::_mm256_sub(y_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        auto y_vec = SIMD_::_mm256_maskloadu(y + i, REMAINDER);
        y_vec = SIMD_::_mm256_sub(y_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<Container ContainerY, typename S>
void SIMD::mul(ContainerY& Y, const S k) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const auto k_vec = SIMD_::_mm256_set1(k);
    for(; i < EDGE; i += 8) {
        auto y_vec = SIMD_::_mm256_loadu(y + i);
        y_vec = SIMD_::_mm256_mul(y_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        auto y_vec = SIMD_::_mm256_maskloadu(y + i, REMAINDER);
        y_vec = SIMD_::_mm256_mul(y_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<Container ContainerY, typename S>
void SIMD::div(ContainerY& Y, const S k) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const float k_ps = static_cast<float>(k);
    const __m256 k_vec = SIMD_::_mm256_set1(k_ps);
    for(; i < EDGE; i += 8) {
        auto y_vec = SIMD_::_mm256_loadu(y + i);
        y_vec = SIMD_::_mm256_div(y_vec, k_vec);

        SIMD_::_mm256_storeu(y + i, y_vec);
    }

    if(REMAINDER != 0) {
        auto y_vec = SIMD_::_mm256_maskloadu(y + i, REMAINDER);
        y_vec = SIMD_::_mm256_div(y_vec, k_vec);

        SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
    }
}

template<Container ContainerA, Container ContainerB, Container ContainerC, Container ContainerY>
void SIMD::fmadd(const ContainerA& A, const ContainerB& B, const ContainerC& C, ContainerY& Y) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using C_traits = container_traits<ContainerC>;
    using Y_traits = container_traits<ContainerY>;

    using S = A_traits::element_type;
    using T = B_traits::element_type;
    using U = C_traits::element_type;
    using V = Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    const U* const c = C.data();
    V* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type && B_traits::container_type == C_traits::container_type) {
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto c_vec = SIMD_::_mm256_loadu(c + i);
            auto y_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, c_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto c_vec = SIMD_::_mm256_maskloadu(c + i, REMAINDER);
            auto y_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, c_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        }
    }
}

template<Container ContainerA, Container ContainerB, Container ContainerC, Container ContainerY>
void SIMD::fmsub(const ContainerA& A, const ContainerB& B, const ContainerC& C, ContainerY& Y) {
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using C_traits = container_traits<ContainerC>;
    using Y_traits = container_traits<ContainerY>;

    using S = A_traits::element_type;
    using T = B_traits::element_type;
    using U = C_traits::element_type;
    using V = Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    const U* const c = C.data();
    V* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(A_traits::container_type == B_traits::container_type && B_traits::container_type == C_traits::container_type) {
        for(; i < EDGE; i += 8) {
            const auto a_vec = SIMD_::_mm256_loadu(a + i);
            const auto b_vec = SIMD_::_mm256_loadu(b + i);
            const auto c_vec = SIMD_::_mm256_loadu(c + i);
            auto y_vec = SIMD_::_mm256_fmsub(a_vec, b_vec, c_vec);

            SIMD_::_mm256_storeu(y + i, y_vec);
        }

        if(REMAINDER != 0) {
            const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
            const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);
            const auto c_vec = SIMD_::_mm256_maskloadu(c + i, REMAINDER);
            auto y_vec = SIMD_::_mm256_fmsub(a_vec, b_vec, c_vec);

            SIMD_::_mm256_maskstoreu(y + i, y_vec, REMAINDER);
        }
    }
}

template<Container ContainerA, Container ContainerB, Container ContainerY>
void SIMD::cross(const ContainerA& A, const ContainerB& B, ContainerY& Y) {
    // this will be assuming A is row-major and B is col-major, A is row vector and B is col-major, or A is row major and B is col vector
    // could implement reverse, but would prob be slower because of load and store latency
    using A_traits = container_traits<ContainerA>;
    using B_traits = container_traits<ContainerB>;
    using Y_traits = container_traits<ContainerY>;

    using S = A_traits::element_type;
    using T = B_traits::element_type;
    using U = Y_traits::element_type;

    const S* const a = A.data();
    const T* const b = B.data();
    U* const y = Y.data();

    if constexpr(A_traits::container_type == container::matrix && B_traits::container_type == container::matrix) {
        const std::size_t N = A.rows();
        const std::size_t K = A.cols();
        const std::size_t M = B.cols();
        
        const std::size_t REMAINDER = K % WIDTH;
        const std::size_t EDGE = K - REMAINDER;

        for(std::size_t i = 0; i < N; ++i) {
            const S* const a_row = a + i*K;

            for(std::size_t j = 0; j < M; ++j) {
                const T* const b_col = b + j*K;

                std::size_t k = 0;
                auto sum_vec = SIMD_::_mm256_setzero<U>();
                for(; k < EDGE; k += WIDTH) {
                    const auto a_vec = SIMD_::_mm256_loadu(a_row + k);
                    const auto b_vec = SIMD_::_mm256_loadu(b_col + k);

                    sum_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, sum_vec);
                }

                if(REMAINDER != 0) {
                    const auto a_vec = SIMD_::_mm256_maskloadu(a_row + k, REMAINDER);
                    const auto b_vec = SIMD_::_mm256_maskloadu(b_col + k, REMAINDER);

                    sum_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, sum_vec);
                }

                y[i*M + j] = SIMD_::_mm256_sum(sum_vec);
            }
        }
    }
    else if constexpr(A_traits::container_type == container::vector) {             // B would be a row-major matrix for gradient
        const std::size_t N = A.size();                                 // B would be col-major for predict
        const std::size_t M = Y.size();                                 // accumulates results
        const std::size_t REMAINDER = M % WIDTH;
        const std::size_t EDGE = M - REMAINDER;

        setzero(Y);
        for(std::size_t i = 0; i < N; ++i) {        // loops through one element of A at a time
            const auto a_vec = SIMD_::_mm256_set1(a[i]);
            const T* const b_i = b + i*M;

            std::size_t j = 0;
            for(; j < EDGE; j += WIDTH) {
                const auto b_vec = SIMD_::_mm256_loadu(b_i + j);
                auto y_vec = SIMD_::_mm256_loadu(y + j);

                y_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, y_vec);

                SIMD_::_mm256_storeu(y + j, y_vec);
            }

            if(REMAINDER != 0) {
                const auto b_vec = SIMD_::_mm256_maskloadu(b_i + j, REMAINDER);
                auto y_vec = SIMD_::_mm256_maskloadu(y + j, REMAINDER);

                y_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, y_vec);

                SIMD_::_mm256_maskstoreu(y + j, y_vec, REMAINDER);
            }
        }
    }
    else if constexpr(B_traits::container_type == container::vector) {             // A would be a column-major matrix (A transposed row major) for gradient
        const std::size_t N = B.size();                                 // A would be row-major for predict
        const std::size_t M = Y.size();                                 // calculates one result at a time
        const std::size_t REMAINDER = N % WIDTH;
        const std::size_t EDGE = N - REMAINDER;

        for(std::size_t j = 0; j < M; ++j) {        // loops through one element of B at a time
            const S* const a_j = a + j*N;

            std::size_t i = 0;
            auto sum_vec = SIMD_::_mm256_setzero<U>();
            for(; i < EDGE; i += WIDTH) {
                const auto a_vec = SIMD_::_mm256_loadu(a_j + i);
                const auto b_vec = SIMD_::_mm256_loadu(b + i);

                sum_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, sum_vec);
            }

            if(REMAINDER != 0) {
                const auto a_vec = SIMD_::_mm256_maskloadu(a_j + i, REMAINDER);
                const auto b_vec = SIMD_::_mm256_maskloadu(b + i, REMAINDER);

                sum_vec = SIMD_::_mm256_fmadd(a_vec, b_vec, sum_vec);
            }

            y[j] = SIMD_::_mm256_sum(sum_vec);
        }
    }
}

template<typename T>
float SIMD::sqsum(const Vector<T>& A) {
    const T* const a = A.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    __m256 sum_vec = SIMD_::_mm256_setzero<T>();
    for(; i < EDGE; i += WIDTH) {
        const auto a_vec = SIMD_::_mm256_loadu(a + i);
        const auto sq_vec = SIMD_::_mm256_mul(a_vec, a_vec);

        sum_vec = SIMD_::_mm256_add(sum_vec, sq_vec);
    }

    if(REMAINDER != 0) {
        const auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);
        const auto sq_vec = SIMD_::_mm256_mul(a_vec, a_vec);

        sum_vec = SIMD_::_mm256_add(sum_vec, sq_vec);
    }

    return SIMD_::_mm256_sum(sum_vec);
}

template<typename T>
void SIMD::sqsum(const Matrix<T>& A, Vector<T>& Y) {
    const T* const a = A.data();
    T* const y = Y.data();

    const std::size_t N = A.rows();
    const std::size_t M = A.cols();
    
    if(A.axis() == Major::row) {
        const std::size_t REMAINDER = M % WIDTH;
        const std::size_t EDGE = M - REMAINDER;

        for(std::size_t i = 0; i < N; ++i) {
            const T* const a_row = a + i*M;

            std::size_t j = 0;
            for(; j < EDGE; j += WIDTH) {
                const auto a_vec = SIMD_::_mm256_loadu(a_row + j);
                __m256 sum_vec = SIMD_::_mm256_loadu(y + j);

                const auto sq_vec = SIMD_::_mm256_mul(a_vec, a_vec);
                sum_vec = SIMD_::_mm256_add(sum_vec, sq_vec);

                SIMD_::_mm256_storeu(y + j, sum_vec);
            }

            if(REMAINDER != 0) {
                const auto a_vec = SIMD_::_mm256_maskloadu(a_row + j, REMAINDER);
                __m256 sum_vec = SIMD_::_mm256_maskloadu(y + j, REMAINDER);

                const auto sq_vec = SIMD_::_mm256_mul(a_vec, a_vec);
                sum_vec = SIMD_::_mm256_add(sum_vec, sq_vec);

                SIMD_::_mm256_maskstoreu(y + j, sum_vec, REMAINDER);
            }
        }
    }
    else if(A.axis() == Major::col) {
        const std::size_t REMAINDER = N % WIDTH;
        const std::size_t EDGE = N - REMAINDER;

        for(std::size_t j = 0; j < M; ++j) {
            const T* const a_col = a + j*N;

            std::size_t i = 0;
            __m256 sum_vec = SIMD_::_mm256_setzero<T>();
            for(; i < EDGE; i += WIDTH) {
                const auto a_vec = SIMD_::_mm256_loadu(a_col + i);
                const auto sq_vec = SIMD_::_mm256_mul(a_vec, a_vec);

                sum_vec = SIMD_::_mm256_add(sum_vec, sq_vec);
            }

            if(REMAINDER != 0) {
                const auto a_vec = SIMD_::_mm256_maskloadu(a_col + i, REMAINDER);
                const auto sq_vec = SIMD_::_mm256_mul(a_vec, a_vec);

                sum_vec = SIMD_::_mm256_add(sum_vec, sq_vec);
            }

            y[j] = SIMD_::_mm256_sum(sum_vec);
        }
    }
}

/*
template<typename T>
float SIMD::lpnorm(const float p, const Vector<T>& A) {
    const T* const a = A.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const __m256 p_vec = SIMD_::_mm256_set1(p);
    __m256 sum_vec = SIMD_::_mm256_setzero<float>();
    for(; i < EDGE; i += 8) {
        auto a_vec = SIMD_::_mm256_loadu(a + i);

        a_vec = SIMD_::_mm256_abs(a_vec);
        const __m256 pow_vec = SIMD_::_mm256_pow(a_vec, p_vec);
        sum_vec = SIMD_::_mm256_add(pow_vec, sum_vec);
    }

    if(REMAINDER != 0) {
        auto a_vec = SIMD_::_mm256_maskloadu(a + i, REMAINDER);

        a_vec = SIMD_::_mm256_abs(a_vec);
        const __m256 pow_vec = SIMD_::_mm256_pow(a_vec, p_vec);
        sum_vec = SIMD_::_mm256_add(pow_vec, sum_vec);
    }

    const float sum = SIMD_::_mm256_sum(sum_vec);
    sum_vec = SIMD_::_mm256_set1(sum);

    const __m256 rcp_vec = SIMD_::_mm256_rcp(p_vec);
    const __m256 norm_vec = SIMD_::_mm256_pow(sum_vec, rcp_vec);

    return _mm256_cvtss_f32(sum_vec);
}

template<typename T>
void SIMD::lpnorm(const float, const Matrix<T>&, Vector<float>&);
*/

// void SIMD::softmax(const Matrix<float>&, const Vector<int>&, Matrix<float>&);