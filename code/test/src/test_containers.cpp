#include <gtest/gtest.h>

#include "interface/containers.h"

TEST(VectorTestInt, Constructors) {
    const std::size_t N = 12;
    Wally::Vector<int> v1(N);

    ASSERT_EQ(N, v1.size());

    const int INIT = 7;
    Wally::Vector<int> v2(N, INIT);

    ASSERT_EQ(N, v2.size());

    const int* const v2_data = v2.data();
    for(std::size_t i = 0; i < N; ++i)
        ASSERT_EQ(INIT, v2_data[i]);
}

TEST(VectorTestInt, RuleOfThree) {
    const std::size_t N = 12;
    Wally::Vector<int> v(N);

    int* const v_data = v.data();
    for(std::size_t i = 0; i < N; ++i)
        v_data[i] = static_cast<int>(i);

    Wally::Vector<int> v1(v);

    ASSERT_EQ(v.size(), v1.size());

    const int* const v1_data = v1.data();
    for(std::size_t i = 0; i < N; ++i)
        ASSERT_EQ(v_data[i], v1_data[i]);

    Wally::Vector<int> v2 = v;

    ASSERT_EQ(v.size(), v2.size());

    const int* const v2_data = v2.data();
    for(std::size_t i = 0; i < N; ++i)
        ASSERT_EQ(v_data[i], v2_data[i]);
}

TEST(VectorTestInt, AccessOperator) {
    const std::size_t N = 12;
    Wally::Vector<int> v(N);

    for(std::size_t i = 0; i < N; ++i)
        v(i) = static_cast<int>(i);

    for(std::size_t i = 0; i < N; ++i)
        ASSERT_EQ(v(i), static_cast<int>(i));
}

TEST(VectorTestInt, ContiguousMemory) { 
    const std::size_t N = 12;
    Wally::Vector<int> v(N);

    int* const v_data = v.data();

    for(std::size_t i = 0; i < N; ++i)
        v(i) = static_cast<int>(i);

    for(std::size_t i = 0; i < N; ++i)
        ASSERT_EQ(*(v_data + i), static_cast<int>(i));
}

TEST(VectorTestFloat, Constructors) {
    const std::size_t N = 12;
    Wally::Vector<float> v1(N);

    ASSERT_EQ(N, v1.size());

    const float INIT = 7.0f;
    Wally::Vector<float> v2(N, INIT);

    ASSERT_EQ(N, v2.size());

    const float* const v2_data = v2.data();
    for(std::size_t i = 0; i < N; ++i)
        ASSERT_EQ(INIT, v2_data[i]);
}

TEST(VectorTestFloat, RuleOfThree) {
    const std::size_t N = 12;
    Wally::Vector<float> v(N);

    float* const v_data = v.data();
    for(std::size_t i = 0; i < N; ++i)
        v_data[i] = static_cast<float>(i);

    Wally::Vector<float> v1(v);

    ASSERT_EQ(v.size(), v1.size());

    const float* const v1_data = v1.data();
    for(std::size_t i = 0; i < N; ++i)
        ASSERT_FLOAT_EQ(v_data[i], v1_data[i]);

    Wally::Vector<float> v2 = v;

    ASSERT_EQ(v.size(), v2.size());

    const float* const v2_data = v2.data();
    for(std::size_t i = 0; i < N; ++i)
        ASSERT_FLOAT_EQ(v_data[i], v2_data[i]);
}

TEST(VectorTestFloat, AccessOperator) {
    const std::size_t N = 12;
    Wally::Vector<float> v(N);

    for(std::size_t i = 0; i < N; ++i)
        v(i) = static_cast<float>(i);

    for(std::size_t i = 0; i < N; ++i)
        ASSERT_FLOAT_EQ(v(i), static_cast<float>(i));
}

TEST(VectorTestFloat, ContiguousMemory) { 
    const std::size_t N = 12;
    Wally::Vector<float> v(N);

    float* const v_data = v.data();

    for(std::size_t i = 0; i < N; ++i)
        v(i) = static_cast<float>(i);

    for(std::size_t i = 0; i < N; ++i)
        ASSERT_FLOAT_EQ(*(v_data + i), static_cast<float>(i));
}

TEST(MatrixTestInt, Constructors) {
    const std::size_t N = 3;
    const std::size_t M = 4;
    Wally::Matrix<int> X(N, M);

    ASSERT_EQ(X.rows(), N);
    ASSERT_EQ(X.cols(), M);
    ASSERT_EQ(X.size(), N*M);
    ASSERT_EQ(X.axis(), Wally::Major::row);

    Wally::Matrix<int> Y(N, M, Wally::Major::col);

    ASSERT_EQ(Y.rows(), N);
    ASSERT_EQ(Y.cols(), M);
    ASSERT_EQ(Y.size(), N*M);
    ASSERT_EQ(Y.axis(), Wally::Major::col);

    const int INIT = 7;
    Wally::Matrix<int> Z(N, M, Wally::Major::col, INIT);

    ASSERT_EQ(Z.rows(), N);
    ASSERT_EQ(Z.cols(), M);
    ASSERT_EQ(Z.size(), N*M);
    ASSERT_EQ(Z.axis(), Wally::Major::col);

    const int* const z_data = Z.data();
    for(std::size_t i = 0; i < N*M; ++i)
        ASSERT_EQ(z_data[i], INIT);
}

TEST(MatrixTestInt, RuleOfThree) {
    const std::size_t N = 3;
    const std::size_t M = 4;
    Wally::Matrix<int> X(N, M, Wally::Major::row);

    int* const x_data = X.data();
    for(std::size_t i = 0; i < N*M; ++i)
        x_data[i] = static_cast<int>(i);

    Wally::Matrix<int> X1(X);

    ASSERT_EQ(X.rows(), X1.rows());
    ASSERT_EQ(X.cols(), X1.cols());
    ASSERT_EQ(X.size(), X1.size());
    ASSERT_EQ(X.axis(), X1.axis());

    const int* const x1_data = X1.data();
    for(std::size_t i = 0; i < N*M; ++i)
        ASSERT_EQ(x_data[i], x1_data[i]);

    Wally::Matrix<int> X2(N, M);
    X2 = X;

    ASSERT_EQ(X.rows(), X2.rows());
    ASSERT_EQ(X.cols(), X2.cols());
    ASSERT_EQ(X.size(), X2.size());
    ASSERT_EQ(X.axis(), X2.axis());

    const int* const x2_data = X1.data();
    for(std::size_t i = 0; i < N*M; ++i)
        ASSERT_EQ(x_data[i], x2_data[i]);
}

TEST(MatrixTestInt, AccessOperator) {
    const std::size_t N = 3;
    const std::size_t M = 4;
    Wally::Matrix<int> X(N, M);

    for(std::size_t i = 0; i < N; ++i) {
        for(std::size_t j = 0; j < M; ++j)
            X(i, j) = static_cast<int>(i*M + j);
    }

    for(std::size_t i = 0; i < N; ++i) {
        for(std::size_t j = 0; j < M; ++j)
            ASSERT_EQ(X(i, j), static_cast<int>(i*M + j));
    }
}

TEST(MatrixTestInt, ContiguousMemory) {
    const std::size_t N = 3;
    const std::size_t M = 4;
    Wally::Matrix<int> X_row(N, M, Wally::Major::row);

    for(std::size_t i = 0; i < N; ++i) {
        for(std::size_t j = 0; j < M; ++j)
            X_row(i, j) = static_cast<int>(i*M + j);
    }

    const int* const x_row_data = X_row.data();
    for(std::size_t i = 0; i < N*M; ++i)
        ASSERT_EQ(*(x_row_data + i), static_cast<int>(i));

    Wally::Matrix<int> X_col(N, M, Wally::Major::col);

    for(std::size_t j = 0; j < M; ++j) {
        for(std::size_t i = 0; i < N; ++i)
            X_col(i, j) = static_cast<int>(j*N + i);
    }

    const int* const x_col_data = X_col.data();
    for(std::size_t i = 0; i < N*M; ++i)
        ASSERT_EQ(*(x_col_data + i), static_cast<int>(i));
}

TEST(MatrixTestFloat, Constructors) {
    const std::size_t N = 3;
    const std::size_t M = 4;
    Wally::Matrix<float> X(N, M);

    ASSERT_EQ(X.rows(), N);
    ASSERT_EQ(X.cols(), M);
    ASSERT_EQ(X.size(), N*M);
    ASSERT_EQ(X.axis(), Wally::Major::row);

    Wally::Matrix<float> Y(N, M, Wally::Major::col);

    ASSERT_EQ(Y.rows(), N);
    ASSERT_EQ(Y.cols(), M);
    ASSERT_EQ(Y.size(), N*M);
    ASSERT_EQ(Y.axis(), Wally::Major::col);

    const float INIT = 7.0f;
    Wally::Matrix<float> Z(N, M, Wally::Major::col, INIT);

    ASSERT_EQ(Z.rows(), N);
    ASSERT_EQ(Z.cols(), M);
    ASSERT_EQ(Z.size(), N*M);
    ASSERT_EQ(Z.axis(), Wally::Major::col);

    const float* const z_data = Z.data();
    for(std::size_t i = 0; i < N*M; ++i)
        ASSERT_FLOAT_EQ(z_data[i], INIT);
}

TEST(MatrixTestFloat, RuleOfThree) {
    const std::size_t N = 3;
    const std::size_t M = 4;
    Wally::Matrix<float> X(N, M, Wally::Major::row);

    float* const x_data = X.data();
    for(std::size_t i = 0; i < N*M; ++i)
        x_data[i] = static_cast<float>(i);

    Wally::Matrix<float> X1(X);

    ASSERT_EQ(X.rows(), X1.rows());
    ASSERT_EQ(X.cols(), X1.cols());
    ASSERT_EQ(X.size(), X1.size());
    ASSERT_EQ(X.axis(), X1.axis());

    const float* const x1_data = X1.data();
    for(std::size_t i = 0; i < N*M; ++i)
        ASSERT_FLOAT_EQ(x_data[i], x1_data[i]);

    Wally::Matrix<float> X2(N, M);
    X2 = X;

    ASSERT_EQ(X.rows(), X2.rows());
    ASSERT_EQ(X.cols(), X2.cols());
    ASSERT_EQ(X.size(), X2.size());
    ASSERT_EQ(X.axis(), X2.axis());

    const float* const x2_data = X1.data();
    for(std::size_t i = 0; i < N*M; ++i)
        ASSERT_FLOAT_EQ(x_data[i], x2_data[i]);
}

TEST(MatrixTestFloat, AccessOperator) {
    const std::size_t N = 3;
    const std::size_t M = 4;
    Wally::Matrix<float> X(N, M);

    for(std::size_t i = 0; i < N; ++i) {
        for(std::size_t j = 0; j < M; ++j)
            X(i, j) = static_cast<float>(i*M + j);
    }

    for(std::size_t i = 0; i < N; ++i) {
        for(std::size_t j = 0; j < M; ++j)
            ASSERT_FLOAT_EQ(X(i, j), static_cast<float>(i*M + j));
    }
}

TEST(MatrixTestFloat, ContiguousMemory) {
    const std::size_t N = 3;
    const std::size_t M = 4;
    Wally::Matrix<float> X_row(N, M, Wally::Major::row);

    for(std::size_t i = 0; i < N; ++i) {
        for(std::size_t j = 0; j < M; ++j)
            X_row(i, j) = static_cast<float>(i*M + j);
    }

    const float* const x_row_data = X_row.data();
    for(std::size_t i = 0; i < N*M; ++i)
        ASSERT_FLOAT_EQ(*(x_row_data + i), static_cast<float>(i));

    Wally::Matrix<float> X_col(N, M, Wally::Major::col);

    for(std::size_t j = 0; j < M; ++j) {
        for(std::size_t i = 0; i < N; ++i)
            X_col(i, j) = static_cast<float>(j*N + i);
    }

    const float* const x_col_data = X_col.data();
    for(std::size_t i = 0; i < N*M; ++i)
        ASSERT_FLOAT_EQ(*(x_col_data + i), static_cast<float>(i));
}