#ifndef POLYNOMIAL_REGRESSION_H
#define POLYNOMIAL_REGRESSION_H

#include "linear_regression.h"

namespace Wally {
    class PolynomialRegression : public LinearRegression {
        std::size_t degree;

        std::size_t count_features(const std::size_t);
        std::size_t choose(const std::size_t, const std::size_t);
        Matrix<int> pascal_matrix(const std::size_t, const std::size_t);

    public:
        PolynomialRegression(const std::size_t);

        Matrix<float> transform(const Matrix<float>&);
        
        ~PolynomialRegression() = default;
    };
}

#endif