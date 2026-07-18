#include "../lib/binary_logistic_regression.h"

double BinaryLogisticRegression::mean(const std::vector<double>& x) {
    int points = x.size();

    double avg = 0;
    for(int i = 0; i < points; ++i)
        avg += x[i];
    avg /= points;

    return avg;
}

std::vector<double> BinaryLogisticRegression::mean(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int features = X[0].size();

    std::vector<double> avgs(features, 0);
    for(int j = 0; j < features; ++j) {
        for(int i = 0; i < points; ++i)
            avgs[j] += X[i][j];
        avgs[j] /= points;
    }

    return avgs;
}

double BinaryLogisticRegression::standard_deviation(const std::vector<double>& x) {
    int points = x.size();

    double avg = mean(x);
    double std_dev = 0;
    for(int i = 0; i < points; ++i)
        std_dev += pow(x[i] - avg, 2);
    std_dev = sqrtf(std_dev / points);

    return std_dev;
}

std::vector<double> BinaryLogisticRegression::standard_deviation(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int features = X[0].size();

    std::vector<double> avgs = mean(X);
    std::vector<double> std_devs(features, 0);
    for(int j = 0; j < features; ++j) {
        for(int i = 0; i < points; ++i)
            std_devs[j] += pow(X[i][j] - avgs[j], 2);
        std_devs[j] = sqrtf(std_devs[j] / points);
    }
    
    return std_devs;
}

double BinaryLogisticRegression::z(const std::vector<double>& x) {
    int parameters = weights.size();

    double z = weights[0];
    for(int j = 1; j < parameters; ++j)
        z += weights[j]*x[j-1];

    return z;
}

std::vector<double> BinaryLogisticRegression::z(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int parameters = weights.size();
    
    std::vector<double> z(points, 0);
    for(int i = 0; i < points; ++i)
        z[i] = this->z(X[i]);

    return z;
}

double BinaryLogisticRegression::sigmoid(const std::vector<double>& x) {}
std::vector<double> BinaryLogisticRegression::sigmoid(const std::vector<std::vector<double>>&) {}

std::vector<double> BinaryLogisticRegression::compute_errors(const std::vector<double>&, const std::vector<double>&) {}
double BinaryLogisticRegression::log_loss(const std::vector<double>&, const std::vector<double>&) {}
std::vector<double> BinaryLogisticRegression::compute_gradient(const std::vector<std::vector<double>>&, const std::vector<double>&, const std::vector<double>&) {}
void BinaryLogisticRegression::gradient_descent(const std::vector<double>&, const double&) {}