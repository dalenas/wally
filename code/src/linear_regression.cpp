#include "../lib/linear_regression.h"

std::vector<double> LinearRegression::mean(std::vector<std::vector<double>> X) {
    int points = X.size();
    int dimension = X[0].size();

    std::vector<double> mean(dimension, 0);
    for(int i = 0; i < dimension; ++i) {
        for(int point = 0; point < points; ++points)
            mean[i] += X[point][i];
        mean[i] /= points;
    }

    return mean;
}

double LinearRegression::mean(std::vector<double> x) {
    int points = x.size();

    double mean = 0;
    for(int point = 0; point < points; ++points)
        mean += x[point];
    
    mean /= points;
    return mean;
}

std::vector<double> LinearRegression::standard_deviation(std::vector<std::vector<double>> X) {
    int points = X.size();
    int dimension = X[0].size();

    std::vector<double> mean = this->mean(X);
    std::vector<double> standard_deviation(dimension, 0);
    for(int i = 0; i < dimension; ++i) {
        for(int point = 0; point < points; ++point)
            standard_deviation[i] += pow(X[point][i] - mean[i], 2);
        standard_deviation[i] = sqrtf(standard_deviation[i] / points);
    }
    
    return standard_deviation;
}

double LinearRegression::standard_deviation(std::vector<double> x) {
    int points = x.size();

    double mean = this->mean(x);
    double standard_deviation = 0;
    for(int point = 0; point < points; ++point)
        standard_deviation += pow(x[point] - mean, 2);
    standard_deviation = sqrtf(standard_deviation / points);

    return standard_deviation;
}

double LinearRegression::mean_squared_error(std::vector<double> errors) {
    int points = errors.size();

    double mse = 0;
    for(int point = 0; point < points; ++point)
        mse += pow(errors[point], 2);
    mse /= points;
    
    return mse;
}

std::vector<double> LinearRegression::compute_gradient(std::vector<double> errors, std::vector<std::vector<double>> X) {
    int points = X.size();
    int parameters = weights.size();

    std::vector<double> gradient(parameters, 0);
    for(int point = 0; point < points; ++point)
        gradient[0] += errors[point];
    gradient[0] *= 2 / points;
    for(int i = 1; i < parameters; ++i) {
        for(int point = 0; point < points; ++point)
            gradient[i] += errors[point]*X[point][i-1];
        gradient[i] *= 2 / points;
    }

    return gradient;
}

void LinearRegression::gradient_descent(std::vector<double> gradient, double learning_rate) {
    int parameters = weights.size();

    for(int i = 0; i < parameters; ++i)
        weights[i] -= learning_rate*gradient[i];
}

void LinearRegression::fit(std::vector<std::vector<double>> X) {
    int parameters = X[0].size() + 1;
    weights.reserve(parameters);

    for(int i = 0; i < parameters; ++i)
        weights.push_back(0);
}

void train(std::vector<std::vector<double>>, std::vector<double>, double, double, int);
std::vector<double> predict(std::vector<std::vector<double>>);
void normalize(std::vector<std::vector<double>>);
void normalize(std::vector<double>);