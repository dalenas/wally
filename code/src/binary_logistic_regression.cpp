#include "../lib/binary_logistic_regression.h"

double BinaryLogisticRegression::z(const vector<double>& x) {
    std::size_t parameters = weights.size();

    double z = weights[BIAS];
    for(std::size_t j = 1; j < parameters; ++j)
        z += weights[j]*x[j-1];

    return z;
}

double BinaryLogisticRegression::sigmoid(const vector<double>& x) {
    return 1 / (1 + exp(-z(x)));
}

vector<double> BinaryLogisticRegression::sigmoid(const matrix<double>& X) {
    std::size_t points = X.size();

    vector<double> sigmoids(points, 0);
    for(std::size_t i = 0; i < points; ++i)
        sigmoids[i] = sigmoid(X[i]);

    return sigmoids;
}

vector<double> BinaryLogisticRegression::compute_errors(const vector<int>& y, const vector<double>& p) {
    std::size_t points = y.size();

    vector<double> errors(points, 0);
    for(std::size_t i = 0; i < points; ++i)
        errors[i] = p[i] - y[i];

    return errors;
}

double BinaryLogisticRegression::log_loss(const vector<int>& y, const vector<double>& p) {
    std::size_t points = y.size();

    double loss = 0;
    for(std::size_t i = 0; i < points; ++i)
        loss += -y[i]*log(p[i]) - (1 - y[i])*log(1 - p[i]);
    
    return loss;
}

vector<double> BinaryLogisticRegression::compute_gradient(const matrix<double>& X, const vector<int>& y, const vector<double>& p) {
    std::size_t points = X.size();
    std::size_t parameters = weights.size();

    vector<double> errors = compute_errors(y, p);
    vector<double> gradient(parameters, 0);
    for(std::size_t i = 0; i < points; ++i)
        gradient[BIAS] += errors[i];
    gradient[BIAS] /= points;
    for(std::size_t j = 1; j < parameters; ++j) {
        for(std::size_t i = 0; i < points; ++i)
            gradient[j] += errors[i]*X[i][j-1];
        gradient[j] /= points;
    }

    return gradient;
}

void BinaryLogisticRegression::gradient_descent(const vector<double>& gradient, const double& learning_rate) {
    std::size_t parameters = gradient.size();

    for(std::size_t j = 0; j < parameters; ++j)
        weights[j] -= learning_rate*gradient[j];
}

void BinaryLogisticRegression::fit(const matrix<double>& X, const vector<int>& y) {
    std::size_t points = y.size();

    int classes = 2;
    for(std::size_t i = 0; i < points; ++i)
        if(y[i] > classes) return;

    std::size_t parameters = X[0].size() + 1;
    weights = vector<double>(parameters, 0);
}

vector<int> BinaryLogisticRegression::predict(const matrix<double>& X) {
    std::size_t points = X.size();

    vector<double> p = sigmoid(X);
    vector<int> y_pred(points, 0);
    for(std::size_t i = 0; i < points; ++i)
        if(p[i] > 0.5) y_pred[i] = 1;
    
    return y_pred;
}

void BinaryLogisticRegression::train(const matrix<double>& X, const vector<int>& y, const double& learning_rate, const double& tol, const std::size_t& max_iter) {
    fit(X, y);
    for(std::size_t k = 0; k < max_iter; ++k) {
        vector<double> p = sigmoid(X);

        if(abs(log_loss(y, p)) < tol)
            return;

        vector<double> gradient = compute_gradient(X, y, p);
        gradient_descent(gradient, learning_rate);
    }
}

void BinaryLogisticRegression::_params() {
    std::size_t parameters = weights.size();

    std::cout << '[' << weights[BIAS];
    for(std::size_t i = 1; i < parameters; ++i)
        std::cout << ", " << weights[i];
    std::cout << ']' << std::endl;
}