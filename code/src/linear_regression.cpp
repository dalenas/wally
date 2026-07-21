#include "../lib/linear_regression.h"

vector<double> LinearRegression::errors(const vector<double>& y_pred, const vector<double>& y) {
    std::size_t points = y.size();

    vector<double> errors(points, 0);
    for(std::size_t point = 0; point < points; ++point)
        errors[point] = y_pred[point] - y[point];
    
    return errors;
}

double LinearRegression::mse_loss(const vector<double>& errors) {
    std::size_t points = errors.size();

    double mse = 0;
    for(std::size_t point = 0; point < points; ++point)
        mse += pow(errors[point], 2);
    mse /= points;
    
    return mse;
}

vector<double> LinearRegression::compute_gradient(const vector<double>& errors, const matrix<double>& X) {
    std::size_t points = X.size();
    std::size_t parameters = weights.size();

    vector<double> gradient(parameters, 0);
    for(std::size_t point = 0; point < points; ++point)
        gradient[BIAS] += errors[point];
    gradient[BIAS] = 2 * gradient[BIAS] / points;
    for(std::size_t i = 1; i < parameters; ++i) {
        for(std::size_t point = 0; point < points; ++point)
            gradient[i] += errors[point]*X[point][i-1];
        gradient[i] = 2 * gradient[i] / points;
    }

    return gradient;
}

void LinearRegression::gradient_descent(const vector<double>& gradient, const double& learning_rate) {
    int parameters = weights.size();

    for(std::size_t i = 0; i < parameters; ++i)
        weights[i] -= learning_rate*gradient[i];
}

void LinearRegression::fit(const matrix<double>& X) {
    int parameters = X[0].size() + 1;
    weights = vector<double>(parameters, 0);
}

void LinearRegression::train(const matrix<double>& X, const vector<double>& y, const double& learning_rate, const double& tol, const std::size_t& max_iter) {
    fit(X);
    for(std::size_t step = 0; step < max_iter; ++step) {
        vector<double> y_pred = predict(X);
        vector<double> y_error = errors(y_pred, y);

        if(mse_loss(y_error) < tol)
            return;
        
        vector<double> gradient = compute_gradient(y_error, X);
        gradient_descent(gradient, learning_rate);
    }
}

vector<double> LinearRegression::predict(const matrix<double>& X) {
    std::size_t points = X.size();
    std::size_t parameters = weights.size();
    vector<double> y_pred(points, weights[0]);

    for(std::size_t point = 0; point < points; ++point) {
        for(std::size_t i = 1; i < parameters; ++i)
            y_pred[point] += weights[i]*X[point][i-1];
    }

    return y_pred;
}

void LinearRegression::_params() {
    std::size_t parameters = weights.size();

    std::cout << '[' << weights[BIAS];
    for(std::size_t i = 1; i < parameters; ++i)
        std::cout << ", " << weights[i];
    std::cout << ']' << std::endl;
}