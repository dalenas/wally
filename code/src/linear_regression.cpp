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