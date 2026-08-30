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

template<typename Container>
void add(const Container::element_type, const Container&, Container&) {
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
void sub(const Container::element_type, const Container&, Container&) {
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
void mul(const Container::element_type, const Container&, Container&) {
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
void div(const Container::element_type, const Container&, Container&) {
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