#include "../../lib/abstract/simd.h"

using namespace Wally;
using namespace Abstract;

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
    using traits = container_traits<Container>;
    using T = typename container_traits<Container>::element_type;

    const T* const a = A.data();
    T* cosnt y = Y.data();

    const std::size_t N = A.size();
    const std::size_t REMAINDER = N % WIDTH;
    const std::size_t EDGE = N - REMAINDER;

    std::size_t i = 0;
    if constexpr(std::is_same_v<T, float> || std::is_same_v<S, float>) {
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
    } else if constexpr(std::is_same_v<S, int> && std::is_same_v<T, int>) {
        __m256i k_vec = _mm256_set1_epi32(k);

        for(; i < EDGE; i += WIDTH) {
            __m256i a_vec = _mm256_loadu_epi32(a + i);
            __m256i y_vec = _mm256_add_epi32(k_vec, a_vec);

            _mm256_storeu_epi32(y + i);
        }

        if(REMAINDER != 0) {
            __m256i a_vec = SIMD_::load_k(a + i, REMAINDER);
            __m256i y_vec = _mm256_add_ps(k_vec, a_vec);

            SIMD_::store_k(y + i, y_vec, REMAINDER);
        }
    }
}

template<typename S, typename Container>
void sub(const S, const Container&, Container&) {
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

template<typename S, typename Container>
void mul(const S, const Container&, Container&) {
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

template<typename S, typename Container>
void div(const S, const Container&, Container&) {
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
void add(const Container&, const Container&, Container&) {
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