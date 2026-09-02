#include "../../lib/abstract/simd.h"

using namespace Wally;
using namespace Abstract;

__m256 SIMD::SIMD_::_mm256_loadu(const float* const addr) { return _mm256_loadu_ps(addr); }
__m256i SIMD::SIMD_::_mm256_loadu(const int* const addr) { return _mm256_loadu_epi32(addr); }

__m256 SIMD::SIMD_::_mm256_maskloadu(const float* const addr, std::size_t k) { return _mm256_maskload_ps(addr, masks[k]); }
__m256i SIMD::SIMD_::_mm256_maskloadu(const int* const addr, std::size_t k) { return _mm256_maskload_epi32(addr, masks[k]); }

void SIMD::SIMD_::_mm256_storeu(float* addr, __m256 a) { _mm256_storeu_ps(addr, a); }
void SIMD::SIMD_::_mm256_storeu(int* addr, __m256i a) { _mm256_storeu_epi32(addr, a); }

void SIMD::SIMD_::_mm256_maskstoreu(float* addr, __m256 a, std::size_t k) { _mm256_maskstore_ps(addr, masks[k], a); }
void SIMD::SIMD_::_mm256_maskstoreu(int* addr, __m256i a, std::size_t k) { _mm256_maskstore_epi32(addr, masks[k], a); }

__m256 SIMD::SIMD_::_mm256_set1(float k) { return _mm256_set1_ps(k); }
__m256 SIMD::SIMD_::_mm256_set1(int k) { return _mm256_set1_ps(static_cast<float>(k)); }            // may want to change to m256i, which would require if constexpr conditions in scalar operations

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

template<typename ContainerY>
void setzero(ContainerY& Y) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(std::is_same_v<U, float>) {
        const __m256 zero = _mm256_setzero_ps();

        for(; i < EDGE; i += WIDTH)
            _mm256_storeu_ps(y + i, zero);

        if(REMAINDER != 0)
            SIMD_::store_k(y + i, zero, REMAINDER);
    } 
    else if constexpr(std::is_same_v<U, int>) {
        const __m256i zero = _mm256_setzero_si256();

        for(; i < EDGE; i += WIDTH)
            _mm256_storeu_si256(y + i, zero);

        if(REMAINDER != 0)
            SIMD_::store_k(y + i, zero, REMAINDER);
    }
}

template<typename Container>
float sum(const Container&);

template<typename T>
void sum(const Matrix<T>&, Vector<T>&);
template<typename T>
T dot(const Vector<T>&, const Vector<T>&);

template<typename S, typename ContainerA, typename ContainerY>
void add(const S k, const ContainerA& A, ContainerY& Y) {
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

template<typename S, typename ContainerA, typename ContainerY>
void sub(const S k, const ContainerA& A, ContainerY& Y) {
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

template<typename ContainerA, typename S, typename ContainerY>
void sub(const ContainerA& A, const S k, ContainerY& Y) {
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

template<typename S, typename ContainerA, typename ContainerY>
void mul(const S k, const ContainerA& A, ContainerY& Y) {
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

template<typename S, typename ContainerA, typename ContainerY>
void div(const S k, const ContainerA& A, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const __m256 k_vec = SIMD_::_mm256_set1(k);
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

template<typename ContainerA, typename S, typename ContainerY>
void div(const ContainerA& A, const S k, ContainerY& Y) {
    using T = typename container_traits<ContainerA>::element_type;
    using U = typename container_traits<ContainerY>::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    const __m256 k_vec = SIMD_::_mm256_set1(k);
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

template<typename ContainerA, typename ContainerB, typename ContainerY>
void add(const ContainerA& A, const ContainerB& B, ContainerY& Y) {
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

template<typename ContainerA, typename ContainerB, typename ContainerY>
void sub(const ContainerA&, const ContainerB&, ContainerY&) {
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

template<typename ContainerA, typename ContainerB, typename ContainerY>
void mul(const ContainerA&, const ContainerB&, ContainerY&) {
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

template<typename ContainerA, typename ContainerB, typename ContainerY>
void div(const ContainerA&, const ContainerB&, ContainerY&) {
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

template<typename ContainerY, typename S>
void add(ContainerY& Y, const S k) {
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

template<typename ContainerY, typename S>
void sub(ContainerY& Y, const S k) {
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

template<typename ContainerY, typename S>
void mul(ContainerY& Y, const S k) {
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

template<typename ContainerY, typename S>
void div(ContainerY& Y, const S k) {
    using U = typename container_traits<ContainerY>::element_type;

    U* const y = Y.data();
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    const auto k_vec = SIMD_::_mm256_set1(k);
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

template<typename ContainerA, typename ContainerB, typename ContainerC, typename ContainerY>
void fmadd(const ContainerA& A, const ContainerB& B, const ContainerC& C, ContainerY& Y) {
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

template<typename ContainerA, typename ContainerB, typename ContainerC, typename ContainerY>
void fmsub(const ContainerA& A, const ContainerB& B, const ContainerC& C, ContainerY& Y) {
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



























template<typename Container>
void SIMD::setzero(Container& Y) {
    using traits = container_traits<Container>;
    using T = typename traits::element_type;

    T* const y = Y.data();
    
    const std::size_t N = Y.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(std::is_same_v<T, float>) {
        const __m256 zero = _mm256_setzero_ps();

        for(; i < EDGE; i += WIDTH)
            _mm256_storeu_ps(y + i, zero);

        if(REMAINDER != 0)
            SIMD_::store_k(y + i, zero, REMAINDER);
    } 
    else if constexpr(std::is_same_v<T, int>) {
        const __m256i zero = _mm256_setzero_si256();

        for(; i < EDGE; i += WIDTH)
            _mm256_storeu_si256(y + i, zero);

        if(REMAINDER != 0)
            SIMD_::store_k(y + i, zero, REMAINDER);
    }
}

template<typename Container>
Container::element_type sum(const Container&) {
    using traits = container_traits<Container>;
    using T = typename container_traits<Container>::element_type;

    if constexpr(traits::container_type == container::vector) {
        if constexpr(std::is_same_v<T, float>) {

        } else if constexpr(std::is_same_v<T, int>) {

        }
    } else if constexpr(traits::container_type == container::matrix) {
        if constexpr(std::is_same_v<T, float>) {
            
        } else if constexpr(std::is_same_v<T, int>) {

        }
    }
}

template<typename T>
void sum(const Matrix<T>&, Vector<T>&) {
    if constexpr(std::is_same_v<T, float>) {

    } else if constexpr(std::is_same_v<T, int>) {

    }
}

template<typename T>
T dot(const Vector<T>&, const Vector<T>&) {}

template<typename S, typename Container>
void add(const S k, const Container& A, Container& Y) {
    using A_traits = container_traits<A>;
    using T = typename A_traits::element_type;
    using Y_traits = container_traits<Y>;
    using U = typename Y_traits::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(std::is_same_v<U, float>) {
        __m256 k_vec = _mm256_set1_ps(k);
        for(; i < EDGE; i += WIDTH) {
            __m256 a_vec = _mm256_loadu_ps(a + i);
            __m256 y_vec = _mm256_add_ps(k_vec, a_vec);

            _mm256_storeu_ps(y + i);
        }

        if(REMAINDER != 0) {
            __m256 a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256 y_vec = _mm256_add_ps(k_vec, a_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    } else if constexpr(std::is_same_v<U, int>) {
        __m256i k_vec = _mm256_set1_epi32(k);
        for(; i < EDGE; i += WIDTH) {
            __m256i a_vec = _mm256_loadu_epi32(a + i);
            __m256i y_vec = _mm256_add_epi32(k_vec, a_vec);

            _mm256_storeu_epi32(y + i);
        }

        if(REMAINDER != 0) {
            __m256i a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256i y_vec = _mm256_add_epi32(k_vec, a_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    }
}

template<typename S, typename Container>
void sub(const S k, const Container& A, Container& Y) {
    using A_traits = container_traits<A>;
    using T = typename A_traits::element_type;
    using Y_traits = container_traits<Y>;
    using U = typename Y_traits::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(std::is_same_v<U, float>) {
        __m256 k_vec = _mm256_set1_ps(k);
        for(; i < EDGE; i += WIDTH) {
            __m256 a_vec = _mm256_loadu_ps(a + i);
            __m256 y_vec = _mm256_sub_ps(k_vec, a_vec);

            _mm256_storeu_ps(y + i);
        }

        if(REMAINDER != 0) {
            __m256 a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256 y_vec = _mm256_sub_ps(k_vec, a_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    } else if constexpr(std::is_same_v<U, int>) {
        __m256i k_vec = _mm256_set1_epi32(k);
        for(; i < EDGE; i += WIDTH) {
            __m256i a_vec = _mm256_loadu_epi32(a + i);
            __m256i y_vec = _mm256_sub_epi32(k_vec, a_vec);

            _mm256_storeu_epi32(y + i);
        }

        if(REMAINDER != 0) {
            __m256i a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256i y_vec = _mm256_sub_epi32(k_vec, a_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    }
}

template<typename Container, typename S>
void sub(const Container& A, const S k, Container& Y) {
    using A_traits = container_traits<A>;
    using T = typename A_traits::element_type;
    using Y_traits = container_traits<Y>;
    using U = typename Y_traits::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(std::is_same_v<U, float>) {
        __m256 k_vec = _mm256_set1_ps(k);
        for(; i < EDGE; i += WIDTH) {
            __m256 a_vec = _mm256_loadu_ps(a + i);
            __m256 y_vec = _mm256_sub_ps(a_vec, k_vec);

            _mm256_storeu_ps(y + i);
        }

        if(REMAINDER != 0) {
            __m256 a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256 y_vec = _mm256_sub_ps(a_vec, k_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    } else if constexpr(std::is_same_v<U, int>) {
        __m256i k_vec = _mm256_set1_epi32(k);
        for(; i < EDGE; i += WIDTH) {
            __m256i a_vec = _mm256_loadu_epi32(a + i);
            __m256i y_vec = _mm256_sub_epi32(a_vec, k_vec);

            _mm256_storeu_epi32(y + i);
        }

        if(REMAINDER != 0) {
            __m256i a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256i y_vec = _mm256_sub_epi32(a_vec, k_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    }
}

template<typename S, typename Container>
void mul(const S k, const Container& A, Container& Y) {
    using A_traits = container_traits<A>;
    using T = typename A_traits::element_type;
    using Y_traits = container_traits<Y>;
    using U = typename Y_traits::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(std::is_same_v<U, float>) {
        __m256 k_vec = _mm256_set1_ps(k);
        for(; i < EDGE; i += WIDTH) {
            __m256 a_vec = _mm256_loadu_ps(a + i);
            __m256 y_vec = _mm256_mul_ps(k_vec, a_vec);

            _mm256_storeu_ps(y + i);
        }

        if(REMAINDER != 0) {
            __m256 a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256 y_vec = _mm256_sub_ps(k_vec, a_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    } else if constexpr(std::is_same_v<U, int>) {
        __m256i k_vec = _mm256_set1_epi32(k);
        for(; i < EDGE; i += WIDTH) {
            __m256i a_vec = _mm256_loadu_epi32(a + i);
            __m256i y_vec = _mm256_sub_epi32(k_vec, a_vec);

            _mm256_storeu_epi32(y + i);
        }

        if(REMAINDER != 0) {
            __m256i a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256i y_vec = _mm256_sub_epi32(k_vec, a_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    }
}

template<typename Container>
void div(const float k, const Container& A, Container& Y) {
    using A_traits = container_traits<Container>;
    using T = typename A_traits::element_type;
    using Y_traits = container_traits<Container>;
    using U = typename Y_traits::element_type;

    const T* const a = A.data();
    U* const y = Y.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(std::is_same_v<U, float>) {
        __m256 k_vec = _mm256_set1_ps(k);
        for(; i < EDGE; i += WIDTH) {
            __m256 a_vec = _mm256_loadu_ps(a + i);
            __m256 y_vec = _mm256_div_ps(k_vec, a_vec);

            _mm256_storeu_ps(y + i);
        }

        if(REMAINDER != 0) {
            __m256 a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256 y_vec = _mm256_div_ps(k_vec, a_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    } else if constexpr(std::is_same_v<U, int>) {
        __m256i k_vec = _mm256_set1_epi32(k);
        for(; i < EDGE; i += WIDTH) {
            __m256i a_vec = _mm256_loadu_epi32(a + i);
            __m256i y_vec = _mm256_div_epi32(k_vec, a_vec);

            _mm256_storeu_epi32(y + i);
        }

        if(REMAINDER != 0) {
            __m256i a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256i y_vec = _mm256_div_epi32(k_vec, a_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    }
}

template<typename Container, typename S>
void div(const Container&, const S, Container&) {
    using traits = container_traits<Container>;
    using T = typename container_traits<Container>::element_type;

    const T* const a = A.data();
    T* const y = Y.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    __m256 k_vec = _mm256_set1_ps(k);
    for(; i < EDGE; i += WIDTH) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 y_vec = _mm256_div_ps(a_vec, k_vec);

        _mm256_storeu_ps(y + i);
    }

    if(REMAINDER != 0) {
        __m256 a_vec = SIMD_::load_k(a + i, REMAINDER);
        __m256 y_vec = _mm256_div_ps(a_vec, k_vec);

        SIMD_::store_k(y + i, y_vec, REMAINDER);
    }
}

template<typename ContainerA, typename ContainerB, typename ContainerY>
void add(const ContainerA&, const ContainerB&, ContainerY&) {
    using traits = container_traits<Container>;
    using T = typename container_traits<Container>::element_type;

    if constexpr(traits::container_type == container::vector) {
        if constexpr(std::is_same_v<T, float>) {

        } else if constexpr(std::is_same_v<T, int>) {

        }
    } else if constexpr(traits::container_type == container::matrix) {
        if constexpr(std::is_same_v<T, float>) {
            
        } else if constexpr(std::is_same_v<T, int>) {

        }
    }
}

template<typename Container>
void sub(const Container&, const Container&, Container&) {
    using traits = container_traits<Container>;
    using T = typename container_traits<Container>::element_type;

    if constexpr(traits::container_type == container::vector) {
        if constexpr(std::is_same_v<T, float>) {

        } else if constexpr(std::is_same_v<T, int>) {

        }
    } else if constexpr(traits::container_type == container::matrix) {
        if constexpr(std::is_same_v<T, float>) {
            
        } else if constexpr(std::is_same_v<T, int>) {

        }
    }
}

template<typename Container>
void mul(const Container&, const Container&, Container&) {
    using traits = container_traits<Container>;
    using T = typename container_traits<Container>::element_type;

    if constexpr(traits::container_type == container::vector) {
        if constexpr(std::is_same_v<T, float>) {

        } else if constexpr(std::is_same_v<T, int>) {

        }
    } else if constexpr(traits::container_type == container::matrix) {
        if constexpr(std::is_same_v<T, float>) {
            
        } else if constexpr(std::is_same_v<T, int>) {

        }
    }
}

template<typename Container>
void div(const Container&, const Container&, Container&) {
    using traits = container_traits<Container>;
    using T = typename container_traits<Container>::element_type;

    if constexpr(traits::container_type == container::vector) {
        if constexpr(std::is_same_v<T, float>) {

        } else if constexpr(std::is_same_v<T, int>) {

        }
    } else if constexpr(traits::container_type == container::matrix) {
        if constexpr(std::is_same_v<T, float>) {
            
        } else if constexpr(std::is_same_v<T, int>) {

        }
    }
}