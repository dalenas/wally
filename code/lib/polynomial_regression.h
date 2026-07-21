#ifndef POLYNOMIAL_REGRESSION_H
#define POLYNOMIAL_REGRESSION_H

#include "linear_regression.h"

class PolynomialRegression : public LinearRegression {
private:
    unsigned int degree;

    std::size_t count_features(const unsigned int&);
    unsigned int choose(const unsigned int&, const unsigned int&);
    matrix<unsigned int> pascal_matrix(const std::size_t&, const std::size_t&);
public:
    PolynomialRegression(const unsigned int&);
    matrix<double> transform(const matrix<double>&);
    ~PolynomialRegression() = default;
};

#endif