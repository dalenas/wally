#include "../lib/linear_regression.h"

std::vector<double> LinearRegression::errors(const std::vector<double>& y_pred, const std::vector<double>& y) {
    int points = y.size();

    std::vector<double> errors(points, 0);
    for(int point = 0; point < points; ++point)
        errors[point] = y_pred[point] - y[point];
    
    return errors;
}

double LinearRegression::mse_loss(const std::vector<double>& errors) {
    int points = errors.size();

    double mse = 0;
    for(int point = 0; point < points; ++point)
        mse += pow(errors[point], 2);
    mse /= points;
    
    return mse;
}

std::vector<double> LinearRegression::compute_gradient(const std::vector<double>& errors, const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int parameters = weights.size();

    std::vector<double> gradient(parameters, 0);
    for(int point = 0; point < points; ++point)
        gradient[BIAS] += errors[point];
    gradient[BIAS] = 2 * gradient[BIAS] / points;
    for(int i = 1; i < parameters; ++i) {
        for(int point = 0; point < points; ++point)
            gradient[i] += errors[point]*X[point][i-1];
        gradient[i] = 2 * gradient[i] / points;
    }

    return gradient;
}

void LinearRegression::gradient_descent(const std::vector<double>& gradient, const double& learning_rate) {
    int parameters = weights.size();

    for(int i = 0; i < parameters; ++i)
        weights[i] -= learning_rate*gradient[i];
}

void LinearRegression::fit(const std::vector<std::vector<double>>& X) {
    int parameters = X[0].size() + 1;
    weights = std::vector<double>(parameters, 0);
}

void LinearRegression::train(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const double& learning_rate=0.05, const double& tol=1e-6, const std::size_t& max_iter=100) {
    for(int step = 0; step < max_iter; ++step) {
        std::vector<double> y_pred = predict(X);
        std::vector<double> y_error = errors(y_pred, y);

        if(mse_loss(y_error) < tol)
            return;
        
        std::vector<double> gradient = compute_gradient(y_error, X);
        gradient_descent(gradient, learning_rate);
    }
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int parameters = weights.size();
    std::vector<double> y_pred(points, weights[0]);

    for(int point = 0; point < points; ++point) {
        for(int i = 1; i < parameters; ++i)
            y_pred[point] += weights[i]*X[point][i-1];
    }

    return y_pred;
}

void LinearRegression::_params() {
    int params = weights.size()-1;

    std::cout << '[';
    for(int i = 0; i < params; ++i)
        std::cout << weights[i] << ", ";
    std::cout << weights[params] << ']';
}