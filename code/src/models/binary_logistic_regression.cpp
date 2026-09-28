#include "models/binary_logistic_regression.h"

void BinaryRegression::z(const Matrix<float>& X) {
    if(X.axis() == Major::row)
        Abstract::SIMD::cross(X, weights, workbench.logits);
    else
        Abstract::SIMD::cross(weights, X, workbench.logits);
    Abstract::SIMD::add(workbench.logits, bias);
}

void BinaryRegression::sigmoid() {
    // fused add-div? A / (B + C) or just do reciprocal of A + B
    // fmadd -1 * z + 1
    // then take reciprocal
}

void BinaryRegression::compute_errors(const Vector<int>& y) {
    Abstract::SIMD::sub(workbench.probs, y, workbench.errors);
}

float BinaryRegression::log_loss(const Vector<int>& y) {
    // need another simd for doing log calculations
}

void BinaryRegression::compute_gradient(const Matrix<float>& X) {
    const float K = -2.0f / static_cast<float>(workbench.errors.size());

    if(X.axis() == Major::row)
        Abstract::SIMD::cross(workbench.errors, X, workbench.grad);
    else
        Abstract::SIMD::cross(X, workbench.errors, workbench.grad);
    Abstract::SIMD::mul(workbench.grad, K);

    workbench.grad_b = K * Abtract::SIMD::sum(workbench.errors);
}

void BinaryRegression::gradient_descent(const float alpha) {
    grad_b -= alpha*workbench.grad_b;
    Abstract::SIMD::sub(weights, alpha, workbench.grad);
}

void BinaryRegression::fit(const Matrix<float>& X) {
    bias = 0.0f;
    weights = Vector<float>(X.cols(), 0.0f);
    workbench = Workbench(X.rows(), X.cols());
}

BinaryRegression::BinaryRegression()
    : weights(0), workbench(0, 0) {}

void BinaryRegression::train(const Matrix<float>& X, const Vector<int>& y, const float alpha, const float tol, const std::size_t max_iter) {
    fit(X);
    for(std::size_t iter = 0; iter < max_iter; ++iter) {
        
    }
}

Vector<int> BinaryRegression::predict(const Matrix<float>&) {

}

void BinaryRegression::params() {

}
























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