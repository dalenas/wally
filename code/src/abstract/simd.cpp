#include "abstract/simd.h"

namespace Wally::Abstract::SIMD::SIMD_ {
    __m256 _mm256_maskloadu(const float* const addr, std::size_t k) { return _mm256_maskload_ps(addr, masks[k]); }
    __m256i _mm256_maskloadu(const int* const addr, std::size_t k) { return _mm256_maskload_epi32(addr, masks[k]); }

    void _mm256_maskstoreu(float* addr, __m256 a, std::size_t k) { _mm256_maskstore_ps(addr, masks[k], a); }
    void _mm256_maskstoreu(int* addr, __m256 a, std::size_t k) {
        __m256i ai = _mm256_cvtps_epi32(a);
        _mm256_maskstore_epi32(addr, masks[k], ai);
    }
    void _mm256_maskstoreu(int* addr, __m256i a, std::size_t k) { _mm256_maskstore_epi32(addr, masks[k], a); }

    __m256 _mm256_set1(float k) { return _mm256_set1_ps(k); }
    __m256i _mm256_set1(int k) { return _mm256_set1_epi32(k); }            // may want to change to m256i, which would require if constexpr conditions in scalar operations
    
    __m256 _mm256_add(const __m256 a, const __m256 b) { return _mm256_add_ps(a, b); }
    __m256 _mm256_add(__m256 a, __m256i b) { 
        __m256 bf = _mm256_cvtepi32_ps(b);
        return _mm256_add_ps(a, bf);
    }
    __m256 _mm256_add(__m256i a, __m256 b) {
        __m256 af = _mm256_cvtepi32_ps(a);
        return _mm256_add_ps(af, b);
    }
    __m256i _mm256_add(__m256i a, __m256i b) { return _mm256_add_epi32(a, b); }

    __m256 _mm256_sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
    __m256 _mm256_sub(__m256 a, __m256i b) {
        __m256 bf = _mm256_cvtepi32_ps(b);
        return _mm256_sub_ps(a, bf);
    }
    __m256 _mm256_sub(__m256i a, __m256 b) {
        __m256 af = _mm256_cvtepi32_ps(a);
        return _mm256_sub_ps(af, b);
    }
    __m256i _mm256_sub(__m256i a, __m256i b) { return _mm256_sub_epi32(a, b); }

    __m256 _mm256_mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
    __m256 _mm256_mul(__m256 a, __m256i b) {
        __m256 bf = _mm256_cvtepi32_ps(b);
        return _mm256_mul_ps(a, bf);
    }
    __m256 _mm256_mul(__m256i a, __m256 b) {
        __m256 af = _mm256_cvtepi32_ps(a);
        return _mm256_mul_ps(af, b);
    }
    __m256i _mm256_mul(__m256i a, __m256i b) { return _mm256_mullo_epi32(a, b); }

    __m256 _mm256_div(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }
    __m256 _mm256_div(__m256 a, __m256i b) {
        __m256 bf = _mm256_cvtepi32_ps(b);
        return _mm256_div_ps(a, bf);
    }
    __m256 _mm256_div(__m256i a, __m256 b) {
        __m256 af = _mm256_cvtepi32_ps(a);
        return _mm256_div_ps(af, b);
    }
    __m256 _mm256_div(__m256i a, __m256i b) {
        __m256 af = _mm256_cvtepi32_ps(a);
        __m256 bf = _mm256_cvtepi32_ps(b);
        return _mm256_div_ps(af, bf);
    }

    float _mm256_sum(__m256 a) {
        __m128 high = _mm256_castps256_ps128(a);
        __m128 low = _mm256_extractf128_ps(a, 1);

        __m128 sum_vec = _mm_add_ps(high, low);

        sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm_hadd_ps(sum_vec, sum_vec);

        return _mm_cvtss_f32(sum_vec);
    }

    int _mm256_sum(__m256i a) {
        __m128i high = _mm256_castsi256_si128(a);
        __m128i low = _mm256_extracti128_si256(a, 1);

        __m128i sum_vec = _mm_add_epi32(high, low);

        sum_vec = _mm_hadd_epi32(sum_vec, sum_vec);
        sum_vec = _mm_hadd_epi32(sum_vec, sum_vec);

        return _mm_cvtsi128_si32(sum_vec);
    }

    __m256 _mm256_fmadd(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }

    __m256 _mm256_fmadd(__m256 a, __m256 b, __m256i c) {
        __m256 cf = _mm256_cvtepi32_ps(c);
        return _mm256_fmadd_ps(a, b, cf);
    }

    __m256 _mm256_fmadd(__m256 a, __m256i b, __m256 c) {
        __m256 bf = _mm256_cvtepi32_ps(b);
        return _mm256_fmadd_ps(a, bf, c);
    }

    __m256 _mm256_fmadd(__m256 a, __m256i b, __m256i c) {
        __m256 bf = _mm256_cvtepi32_ps(b);
        __m256 cf = _mm256_cvtepi32_ps(c);
        return _mm256_fmadd_ps(a, bf, cf);
    }

    __m256 _mm256_fmadd(__m256i a, __m256 b, __m256 c) {
        __m256 af = _mm256_cvtepi32_ps(a);
        return _mm256_fmadd_ps(af, b, c);
    }

    __m256 _mm256_fmadd(__m256i a, __m256 b, __m256i c) {
        __m256 af = _mm256_cvtepi32_ps(a);
        __m256 cf = _mm256_cvtepi32_ps(c);
        return _mm256_fmadd_ps(af, b, cf);
    }

    __m256 _mm256_fmadd(__m256i a, __m256i b, __m256 c) {
        __m256 af = _mm256_cvtepi32_ps(a);
        __m256 bf = _mm256_cvtepi32_ps(b);
        return _mm256_fmadd_ps(af, bf, c);
    }

    __m256 _mm256_fmadd(__m256i a, __m256i b, __m256i c) { 
        __m256 af = _mm256_cvtepi32_ps(a);
        __m256 bf = _mm256_cvtepi32_ps(b);
        __m256 cf = _mm256_cvtepi32_ps(c);
        return _mm256_fmadd_ps(af, bf, cf);
    }

    __m256 _mm256_fmsub(__m256 a, __m256 b, __m256 c) { return _mm256_fmsub_ps(a, b, c); }

    __m256 _mm256_fmsub(__m256 a, __m256 b, __m256i c) {
        __m256 cf = _mm256_cvtepi32_ps(c);
        return _mm256_fmsub_ps(a, b, cf);
    }

    __m256 _mm256_fmsub(__m256 a, __m256i b, __m256 c) {
        __m256 bf = _mm256_cvtepi32_ps(b);
        return _mm256_fmsub_ps(a, bf, c);
    }

    __m256 _mm256_fmsub(__m256 a, __m256i b, __m256i c) {
        __m256 bf = _mm256_cvtepi32_ps(b);
        __m256 cf = _mm256_cvtepi32_ps(c);
        return _mm256_fmsub_ps(a, bf, cf);
    }

    __m256 _mm256_fmsub(__m256i a, __m256 b, __m256 c) {
        __m256 af = _mm256_cvtepi32_ps(a);
        return _mm256_fmsub_ps(af, b, c);
    }

    __m256 _mm256_fmsub(__m256i a, __m256 b, __m256i c) {
        __m256 af = _mm256_cvtepi32_ps(a);
        __m256 cf = _mm256_cvtepi32_ps(c);
        return _mm256_fmsub_ps(af, b, cf);
    }

    __m256 _mm256_fmsub(__m256i a, __m256i b, __m256 c) {
        __m256 af = _mm256_cvtepi32_ps(a);
        __m256 bf = _mm256_cvtepi32_ps(b);
        return _mm256_fmsub_ps(af, bf, c);
    }

    __m256 _mm256_fmsub(__m256i a, __m256i b, __m256i c) { 
        __m256 af = _mm256_cvtepi32_ps(a);
        __m256 bf = _mm256_cvtepi32_ps(b);
        __m256 cf = _mm256_cvtepi32_ps(c);
        return _mm256_fmsub_ps(af, bf, cf);
    }

    __m256 _mm256_abs(__m256 a) {
        const __m256 sign_vec = _mm256_set1_ps(-0.0f);
        return _mm256_andnot_ps(sign_vec, a);
    }

    __m256i _mm256_abs(__m256i a) {
        return _mm256_abs_epi32(a);
    }

    __m256 _mm256_rcp(__m256 a) {
        return _mm256_rcp_ps(a);
    }

    __m256 _mm256_rcp(__m256i a) {
        __m256 af = _mm256_cvtepi32_ps(a);
        return _mm256_rcp_ps(af);
    }

    __m256 _mm256_sqrt(__m256 a) {
        return _mm256_sqrt_ps(a);
    }

    __m256 _mm256_sqrt(__m256i a) {
        __m256 af = _mm256_cvtepi32_ps(a);
        return _mm256_sqrt_ps(af);
    }
};