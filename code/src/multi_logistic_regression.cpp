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
        softmax[k] = exp(z(x, k));
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

double MultiLogisticRegression::cross_entropy_loss(const vector<int>& y, const matrix<double>& P) {
    std::size_t points = y.size();

    double loss = 0;
    for(std::size_t i = 0; i < points; ++i)
        loss += -log(P[i][y[i]]);
    loss /= points;

    return loss;
}

vector<double> MultiLogisticRegression::compute_errors(const int& y, const vector<double>& p) {
    std::size_t classes = p.size();

    vector<double> errors(classes, 0);
    for(std::size_t k; k < classes; ++k)
        errors[k] = y != k ? p[k] : p[k] - 1;

    return errors;
}

matrix<double> MultiLogisticRegression::compute_errors(const vector<int>& y, const matrix<double>& P) {
    std::size_t classes = P.size();
    std::size_t points = y.size();

    matrix<double> errors(points, vector<double>(classes, 0));
    for(std::size_t i = 0; i < points; ++i)
        errors[i] = compute_errors(y[i], P[i]);

    return errors;
}

matrix<double> MultiLogisticRegression::compute_gradient(const matrix<double>& X, const vector<int>& y, const matrix<double>& P) {
    std::size_t classes = weights.size();
    std::size_t parameters = weights[0].size();
    std::size_t points = X.size();

    matrix<double> errors = compute_errors(y, P);
    matrix<double> gradient(classes, vector<double>(parameters, 0));
    for(std::size_t k = 0; k < classes; ++k) {
        for(std::size_t i = 0; i < points; ++i)
            gradient[k][BIAS] += errors[i][k];
        gradient[k][BIAS] /= points;
    }
    for(std::size_t k = 0; k < classes; ++k) {
        for(std::size_t j = 1; j < parameters; ++j) {
            for(std::size_t i = 0; i < points; ++i)
                gradient[k][j] += X[i][j-1]*errors[i][k];
            gradient[k][j] /= points;
        }
    }

    return gradient;
}

void MultiLogisticRegression::gradient_descent(const matrix<double>& X, const vector<int>& y, const matrix<double>& P, const double& learning_rate) {
    std::size_t classes = weights.size();
    std::size_t parameters = weights[0].size();

    matrix<double> gradient = compute_gradient(X, y, P);
    for(std::size_t k = 0; k < classes; ++k) {
        for(std::size_t j = 0; j < parameters; ++j)
            weights[k][j] -= learning_rate*gradient[k][j];
    }
}