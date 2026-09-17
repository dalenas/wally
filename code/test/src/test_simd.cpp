#include <gtest/gtest.h>

#include "abstract/simd.h"

using namespace Wally;
using namespace Abstract;

TEST(SIMDTestsPrivate, LoadStore) {
    const std::size_t N = 10;
    const std::size_t REMAINDER = N % SIMD::WIDTH;

    const float a1[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const float* const p1 = &a1[0];

    auto a1_vec = SIMD::SIMD_::_mm256_loadu(p1);
    auto a1_rem = SIMD::SIMD_::_mm256_maskloadu(p1 + SIMD::WIDTH, REMAINDER);

    float b1[N];

    SIMD::SIMD_::_mm256_storeu(b1, a1_vec);
    SIMD::SIMD_::_mm256_maskstoreu(b1 + SIMD::WIDTH, a1_rem, REMAINDER);

    for(std::size_t i = 0; i < 10; ++i)
        ASSERT_FLOAT_EQ(a1[i], b1[i]);

    const int a2[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const int* const p2 = &a2[0];

    auto a2_vec = SIMD::SIMD_::_mm256_loadu(p2);
    auto a2_rem = SIMD::SIMD_::_mm256_maskloadu(p2 + SIMD::WIDTH, REMAINDER);

    int b2[N];

    SIMD::SIMD_::_mm256_storeu(b2, a2_vec);
    SIMD::SIMD_::_mm256_maskstoreu(b2 + SIMD::WIDTH, a2_rem, REMAINDER);

    for(std::size_t i = 0; i < 10; ++i)
        ASSERT_EQ(a2[i], b2[i]);
}

TEST(SIMDTestsPrivate, Zero) {
    const std::size_t N = 10;
    const std::size_t REMAINDER = N % SIMD::WIDTH;

    float a1[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const float* const p1 = &a1[0];

    auto a1_vec = SIMD::SIMD_::_mm256_loadu(p1);
    auto a1_rem = SIMD::SIMD_::_mm256_maskloadu(p1 + SIMD::WIDTH, REMAINDER);

    a1_vec = SIMD::SIMD_::_mm256_setzero<float>();
    a1_rem = SIMD::SIMD_::_mm256_setzero<float>();

    SIMD::SIMD_::_mm256_storeu(a1, a1_vec);
    SIMD::SIMD_::_mm256_maskstoreu(a1 + SIMD::WIDTH, a1_rem, REMAINDER);

    for(std::size_t i = 0; i < 10; ++i)
        ASSERT_FLOAT_EQ(a1[i], 0.0f);

    int a2[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const int* const p2 = &a2[0];

    auto a2_vec = SIMD::SIMD_::_mm256_loadu(p2);
    auto a2_rem = SIMD::SIMD_::_mm256_maskloadu(p2 + SIMD::WIDTH, REMAINDER);

    a2_vec = SIMD::SIMD_::_mm256_setzero<int>();
    a2_rem = SIMD::SIMD_::_mm256_setzero<int>();

    SIMD::SIMD_::_mm256_storeu(a2, a2_vec);
    SIMD::SIMD_::_mm256_maskstoreu(a2 + SIMD::WIDTH, a2_rem, REMAINDER);

    for(std::size_t i = 0; i < 10; ++i)
        ASSERT_EQ(a2[i], 0);
}

TEST(SIMDTestsPrivate, Set1) {
    const std::size_t N = 10;
    const std::size_t REMAINDER = N % SIMD::WIDTH;

    float a1[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const float* const p1 = &a1[0];

    auto a1_vec = SIMD::SIMD_::_mm256_loadu(p1);
    auto a1_rem = SIMD::SIMD_::_mm256_maskloadu(p1 + SIMD::WIDTH, REMAINDER);

    const float INITf = 7.0f;
    a1_vec = SIMD::SIMD_::_mm256_set1(INITf);
    a1_rem = SIMD::SIMD_::_mm256_set1(INITf);

    SIMD::SIMD_::_mm256_storeu(a1, a1_vec);
    SIMD::SIMD_::_mm256_maskstoreu(a1 + SIMD::WIDTH, a1_rem, REMAINDER);

    for(std::size_t i = 0; i < 10; ++i)
        ASSERT_FLOAT_EQ(a1[i], 7.0f);

    int a2[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const int* const p2 = &a2[0];

    auto a2_vec = SIMD::SIMD_::_mm256_loadu(p2);
    auto a2_rem = SIMD::SIMD_::_mm256_maskloadu(p2 + SIMD::WIDTH, REMAINDER);

    const int INITi = 7;
    a2_vec = SIMD::SIMD_::_mm256_set1(INITi);
    a2_rem = SIMD::SIMD_::_mm256_set1(INITi);

    SIMD::SIMD_::_mm256_storeu(a2, a2_vec);
    SIMD::SIMD_::_mm256_maskstoreu(a2 + SIMD::WIDTH, a2_rem, REMAINDER);

    for(std::size_t i = 0; i < 10; ++i)
        ASSERT_EQ(a2[i], 7);
}

TEST(SIMDTestsPrivate, Add) {
    const std::size_t N = 10;
    const std::size_t REMAINDER = N % SIMD::WIDTH;

    float a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    int b[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    const float* const p1 = &a[0];
    const int* const p2 = &b[0];

    auto a_vec = SIMD::SIMD_::_mm256_loadu(p1);
    auto a_rem = SIMD::SIMD_::_mm256_maskloadu(p1 + SIMD::WIDTH, REMAINDER);
    auto b_vec = SIMD::SIMD_::_mm256_loadu(p2);
    auto b_rem = SIMD::SIMD_::_mm256_maskloadu(p2 + SIMD::WIDTH, REMAINDER);

    auto y1_vec = SIMD::SIMD_::_mm256_add(a_vec, a_vec);
    auto y1_rem = SIMD::SIMD_::_mm256_add(a_rem, a_rem);
    auto y2_vec = SIMD::SIMD_::_mm256_add(a_vec, b_vec);
    auto y2_rem = SIMD::SIMD_::_mm256_add(a_rem, b_rem);
    auto y3_vec = SIMD::SIMD_::_mm256_add(b_vec, a_vec);
    auto y3_rem = SIMD::SIMD_::_mm256_add(b_rem, a_rem);
    auto y4_vec = SIMD::SIMD_::_mm256_add(b_vec, b_vec);
    auto y4_rem = SIMD::SIMD_::_mm256_add(b_rem, b_rem);

    float y1[N];
    float y2[N];
    float y3[N];
    int y4[N];
    SIMD::SIMD_::_mm256_storeu(y1, y1_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y1 + SIMD::WIDTH, y1_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y2, y2_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y2 + SIMD::WIDTH, y2_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y3, y3_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y3 + SIMD::WIDTH, y3_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y4, y4_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y4 + SIMD::WIDTH, y4_rem, REMAINDER);

    for(std::size_t i = 0; i < N; ++i) {
        ASSERT_FLOAT_EQ(y1[i], 2*(static_cast<float>(i)+1));
        ASSERT_FLOAT_EQ(y2[i], 2*(static_cast<float>(i)+1));
        ASSERT_FLOAT_EQ(y3[i], 2*(static_cast<float>(i)+1));
        ASSERT_EQ(y4[i], 2*(static_cast<int>(i)+1));
    }
}

TEST(SIMDTestsPrivate, Sub) {
    const std::size_t N = 10;
    const std::size_t REMAINDER = N % SIMD::WIDTH;

    float a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    int b[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    const float* const p1 = &a[0];
    const int* const p2 = &b[0];

    auto a_vec = SIMD::SIMD_::_mm256_loadu(p1);
    auto a_rem = SIMD::SIMD_::_mm256_maskloadu(p1 + SIMD::WIDTH, REMAINDER);
    auto b_vec = SIMD::SIMD_::_mm256_loadu(p2);
    auto b_rem = SIMD::SIMD_::_mm256_maskloadu(p2 + SIMD::WIDTH, REMAINDER);

    auto y1_vec = SIMD::SIMD_::_mm256_sub(a_vec, a_vec);
    auto y1_rem = SIMD::SIMD_::_mm256_sub(a_rem, a_rem);
    auto y2_vec = SIMD::SIMD_::_mm256_sub(a_vec, b_vec);
    auto y2_rem = SIMD::SIMD_::_mm256_sub(a_rem, b_rem);
    auto y3_vec = SIMD::SIMD_::_mm256_sub(b_vec, a_vec);
    auto y3_rem = SIMD::SIMD_::_mm256_sub(b_rem, a_rem);
    auto y4_vec = SIMD::SIMD_::_mm256_sub(b_vec, b_vec);
    auto y4_rem = SIMD::SIMD_::_mm256_sub(b_rem, b_rem);

    float y1[N];
    float y2[N];
    float y3[N];
    int y4[N];
    SIMD::SIMD_::_mm256_storeu(y1, y1_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y1 + SIMD::WIDTH, y1_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y2, y2_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y2 + SIMD::WIDTH, y2_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y3, y3_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y3 + SIMD::WIDTH, y3_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y4, y4_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y4 + SIMD::WIDTH, y4_rem, REMAINDER);

    for(std::size_t i = 0; i < N; ++i) {
        ASSERT_FLOAT_EQ(y1[i], 0.0f);
        ASSERT_FLOAT_EQ(y2[i], 0.0f);
        ASSERT_FLOAT_EQ(y3[i], 0.0f);
        ASSERT_EQ(y4[i], 0);
    }
}

TEST(SIMDTestsPrivate, Mul) {
    const std::size_t N = 10;
    const std::size_t REMAINDER = N % SIMD::WIDTH;

    float a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    int b[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    const float* const p1 = &a[0];
    const int* const p2 = &b[0];

    auto a_vec = SIMD::SIMD_::_mm256_loadu(p1);
    auto a_rem = SIMD::SIMD_::_mm256_maskloadu(p1 + SIMD::WIDTH, REMAINDER);
    auto b_vec = SIMD::SIMD_::_mm256_loadu(p2);
    auto b_rem = SIMD::SIMD_::_mm256_maskloadu(p2 + SIMD::WIDTH, REMAINDER);

    auto y1_vec = SIMD::SIMD_::_mm256_mul(a_vec, a_vec);
    auto y1_rem = SIMD::SIMD_::_mm256_mul(a_rem, a_rem);
    auto y2_vec = SIMD::SIMD_::_mm256_mul(a_vec, b_vec);
    auto y2_rem = SIMD::SIMD_::_mm256_mul(a_rem, b_rem);
    auto y3_vec = SIMD::SIMD_::_mm256_mul(b_vec, a_vec);
    auto y3_rem = SIMD::SIMD_::_mm256_mul(b_rem, a_rem);
    auto y4_vec = SIMD::SIMD_::_mm256_mul(b_vec, b_vec);
    auto y4_rem = SIMD::SIMD_::_mm256_mul(b_rem, b_rem);

    float y1[N];
    float y2[N];
    float y3[N];
    int y4[N];
    SIMD::SIMD_::_mm256_storeu(y1, y1_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y1 + SIMD::WIDTH, y1_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y2, y2_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y2 + SIMD::WIDTH, y2_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y3, y3_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y3 + SIMD::WIDTH, y3_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y4, y4_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y4 + SIMD::WIDTH, y4_rem, REMAINDER);

    for(std::size_t i = 0; i < N; ++i) {
        float i_sqf = static_cast<float>(i+1)*static_cast<float>(i+1);
        int i_sqi = static_cast<int>(i+1)*static_cast<int>(i+1);
        ASSERT_FLOAT_EQ(y1[i], i_sqf);
        ASSERT_FLOAT_EQ(y2[i], i_sqf);
        ASSERT_FLOAT_EQ(y3[i], i_sqf);
        ASSERT_EQ(y4[i], i_sqi);
    }
}

TEST(SIMDTestsPrivate, Div) {
    const std::size_t N = 10;
    const std::size_t REMAINDER = N % SIMD::WIDTH;

    float a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    int b[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    const float* const p1 = &a[0];
    const int* const p2 = &b[0];

    auto a_vec = SIMD::SIMD_::_mm256_loadu(p1);
    auto a_rem = SIMD::SIMD_::_mm256_maskloadu(p1 + SIMD::WIDTH, REMAINDER);
    auto b_vec = SIMD::SIMD_::_mm256_loadu(p2);
    auto b_rem = SIMD::SIMD_::_mm256_maskloadu(p2 + SIMD::WIDTH, REMAINDER);

    auto y1_vec = SIMD::SIMD_::_mm256_div(a_vec, a_vec);
    auto y1_rem = SIMD::SIMD_::_mm256_div(a_rem, a_rem);
    auto y2_vec = SIMD::SIMD_::_mm256_div(a_vec, b_vec);
    auto y2_rem = SIMD::SIMD_::_mm256_div(a_rem, b_rem);
    auto y3_vec = SIMD::SIMD_::_mm256_div(b_vec, a_vec);
    auto y3_rem = SIMD::SIMD_::_mm256_div(b_rem, a_rem);
    auto y4_vec = SIMD::SIMD_::_mm256_div(b_vec, b_vec);
    auto y4_rem = SIMD::SIMD_::_mm256_div(b_rem, b_rem);

    float y1[N];
    float y2[N];
    float y3[N];
    float y4[N];
    SIMD::SIMD_::_mm256_storeu(y1, y1_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y1 + SIMD::WIDTH, y1_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y2, y2_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y2 + SIMD::WIDTH, y2_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y3, y3_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y3 + SIMD::WIDTH, y3_rem, REMAINDER);
    SIMD::SIMD_::_mm256_storeu(y4, y4_vec);
    SIMD::SIMD_::_mm256_maskstoreu(y4 + SIMD::WIDTH, y4_rem, REMAINDER);

    for(std::size_t i = 0; i < N; ++i) {
        ASSERT_FLOAT_EQ(y1[i], 1.0f);
        ASSERT_FLOAT_EQ(y2[i], 1.0f);
        ASSERT_FLOAT_EQ(y3[i], 1.0f);
        ASSERT_FLOAT_EQ(y4[i], 1.0f);
    }
}
/*
TEST(SIMDTestsPrivate, Sum);
TEST(SIMDTestsPrivate, Fmadd);
TEST(SIMDTestsPrivate, Fmsub);
TEST(SIMDTestsPrivate, Abs);
TEST(SIMDTestsPrivate, Rcp);

TEST(SIMDTests, Zero);
TEST(SIMDTests, Sum);
TEST(SIMDTests, Dot);
TEST(SIMDTests, Add);
TEST(SIMDTests, Sub);
TEST(SIMDTests, Mul);
TEST(SIMDTests, Div);
TEST(SIMDTests, Fmadd);
TEST(SIMDTests, Fmsub);
TEST(SIMDTests, Cross);
TEST(SIMDTests, Sqsum);
*/