#include "../lib/polynomial_regression.h"

PolynomialRegression::PolynomialRegression(const int& d)
    : degree(d) {};

int PolynomialRegression::num_features(const int& dimension) {
    int features = 0;
    for(int k = 1; k <= degree; ++k) {
        int n = dimension + k - 1;
        features += choose(n, k);
        
        /*for(int n_div_k = k+1; n_div_k <= n; ++n_div_k)
            n_choose_k *= n_div_k;
        for(int n_minus_k = n - k; n_minus_k > 1; --n_minus_k)
            n_choose_k /= n_minus_k;
        
        features += n_choose_k;*/
    }
    return features;
}

int PolynomialRegression::choose(const int& n, const int& k) {
    int n_choose_k = 1;

    for(int n_div_k = k+1; n_div_k <= n; ++n_div_k)
        n_choose_k *= n_div_k;
    for(int n_minus_k = n - k; n_minus_k > 1; --n_minus_k)
        n_choose_k /= n_minus_k;

    return n_choose_k;
}

std::vector<std::vector<int>> PolynomialRegression::pascal_matrix(const int& rows, const int& cols) {
    std::vector<std::vector<int>> pascal_matrix(rows, std::vector<int>(cols, 0));
    for(int row = 0; row < rows; ++row) {
        for(int col = 0; col < cols; ++col) {
            int n = cols + row - col;
            int k = cols - col - 1;
            pascal_matrix[row][col] = k > col + 1 ? choose(n, k) : choose(n, row + 1);
        }
    }
    return pascal_matrix;
}

std::vector<std::vector<double>> PolynomialRegression::transform(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int dimension = X[0].size();
    int features = num_features(dimension);

    std::vector<std::vector<int>> P = pascal_matrix(degree-1, dimension);
    std::vector<std::vector<double>> X_transform(points, std::vector<double>(features, 0));
    for(int i = 0; i < dimension; ++i) {
        for(int point = 0; point < points; ++point)
            X_transform[point][i] = X[point][i];
    }
    int row = 0, col = 0, counter = 0;
    int pascal_offset = P[row][col];
    for(int feature = dimension; feature < features; ++feature) {
        int base_feature = col;
        int inductive_feature = feature - pascal_offset;

        for(int point = 0; point < points; ++point)
            X_transform[point][feature] = X_transform[point][base_feature]*X_transform[point][inductive_feature];

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
    return X_transform;
}



/*
int main() {
    int degree = 3;
    int dimension = 3;
    int feature_vectors = 4;
    PolynomialRegression model(degree);
    int features = model.num_features(dimension);
    std::vector<std::vector<double>> X(feature_vectors, std::vector<double>(dimension, 0));

    for(int i = 0; i < feature_vectors; ++i) {
        for(int j = 0; j < dimension; ++j)
            X[i][j] = pow(i, 2) + j;
    }

    std::vector<std::vector<double>> X_transform = model.transform(X);

    for(int i = 0; i < feature_vectors; ++i) {
        for(int j = 0; j < features; ++j)
            std::cout << X_transform[i][j] << ' ';
        std::cout << std::endl;
    }

    return 0;
}
*/