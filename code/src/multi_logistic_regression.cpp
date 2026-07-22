#include "../lib/multi_logistic_regression.h"

double MultiLogisticRegression::z(const vector<double>& x, const int& k) {
    std::size_t parameters = weights[k].size();

    double z = weights[k][BIAS];
    for(std::size_t j = 1; j < parameters; ++j)
        z += weights[k][j]*x[j-1];

    return z;
}

vector<double> MultiLogisticRegression::softmax(const vector<double>& x) {
    std::size_t classes = weights.size();

    double softmax_norm = 0;
    vector<double> softmax(classes, 0);
    for(std::size_t k = 0; k < classes; ++k) {
        softmax[k] = exp(-z(x, k));
        softmax_norm += softmax[k];
    }
    for(std::size_t k = 0; k < classes; ++k)
        softmax[k] /= softmax_norm;

    return softmax;
}

matrix<double> MultiLogisticRegression::softmax(const matrix<double>& X) {
    std::size_t points = X.size();
    std::size_t classes = weights.size();

    matrix<double> softmaxes(points, vector<double>(classes, 0));
    for(std::size_t i = 0; i < points; ++i)
        softmaxes[i] = softmax(X[i]);

    return softmaxes;
}

double MultiLogisticRegression::cross_entropy_loss(const vector<int>&, const vector<double>&);
vector<double> MultiLogisticRegression::compute_errors(const vector<double>&, const vector<double>&);
vector<double> MultiLogisticRegression::compute_gradient(const matrix<double>&, const vector<double>&, const vector<double>&);
void MultiLogisticRegression::gradient_descent(const matrix<double>&, const vector<double>&, const vector<double>&, const double&);