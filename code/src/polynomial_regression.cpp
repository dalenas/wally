#include "../lib/polynomial_regression.h"

PolynomialRegression::PolynomialRegression(const unsigned int& d)
    : degree(d) {};

std::size_t PolynomialRegression::count_features(const unsigned int& dimension) {
    std::size_t features = 0;
    for(unsigned int k = 1; k <= degree; ++k) {
        unsigned int n = dimension + k - 1;
        features += choose(n, k);
    }
    return features;
}

unsigned int PolynomialRegression::choose(const unsigned int& n, const unsigned int& k) {
    unsigned int n_choose_k = 1;

    for(unsigned int n_div_k = k + 1; n_div_k <= n; ++n_div_k)
        n_choose_k *= n_div_k;
    for(unsigned int n_minus_k = n - k; n_minus_k > 1; --n_minus_k)
        n_choose_k /= n_minus_k;

    return n_choose_k;
}

matrix<unsigned int> PolynomialRegression::pascal_matrix(const std::size_t& rows, const std::size_t& cols) {
    matrix<unsigned int> pascal_matrix(rows, vector<unsigned int>(cols, 0));
    for(std::size_t row = 0; row < rows; ++row) {
        for(std::size_t col = 0; col < cols; ++col) {
            unsigned int n = cols + row - col;
            unsigned int k = cols - col - 1;
            pascal_matrix[row][col] = k > col + 1 ? choose(n, k) : choose(n, row + 1);
        }
    }
    return pascal_matrix;
}

matrix<double> PolynomialRegression::transform(const matrix<double>& X) {
    std::size_t points = X.size();
    unsigned int dimension = X[0].size();
    std::size_t features = count_features(dimension);

    matrix<unsigned int> P = pascal_matrix(degree-1, dimension);
    matrix<double> transform(points, vector<double>(features, 0));
    for(std::size_t i = 0; i < dimension; ++i) {
        for(std::size_t point = 0; point < points; ++point)
            transform[point][i] = X[point][i];
    }
    std::size_t row = 0, col = 0, counter = 0;
    unsigned int pascal_offset = P[row][col];
    for(std::size_t feature = dimension; feature < features; ++feature) {
        std::size_t base_feature = col;
        std::size_t inductive_feature = feature - pascal_offset;

        for(std::size_t point = 0; point < points; ++point)
            transform[point][feature] = transform[point][base_feature]*transform[point][inductive_feature];

        ++counter;
        if(counter == P[row][col]) {
            counter = 0;
            col = (col + 1) % P[row].size();
            if(col != 0)
                pascal_offset += P[row][col];
            else {
                row = (row + 1) % P.size();
                pascal_offset = P[row][col];
            }
        }
    }
    return transform;
}