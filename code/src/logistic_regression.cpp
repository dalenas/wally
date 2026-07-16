#include "../lib/logistic_regression.h"

std::vector<double> LogisticRegression::mean(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int dimension = X[0].size();

    std::vector<double> avgs(dimension, 0);
    for(int i = 0; i < dimension; ++i) {
        for(int point = 0; point < points; ++point)
            avgs[i] += X[point][i];
        avgs[i] /= points;
    }

    return avgs;
}

double LogisticRegression::mean(const std::vector<double>& x) {
    int points = x.size();

    double avg = 0;
    for(int point = 0; point < points; ++point)
        avg += x[point];
    
    avg /= points;
    return avg;
}

std::vector<double> LogisticRegression::standard_deviation(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int dimension = X[0].size();

    std::vector<double> avgs = mean(X);
    std::vector<double> std_devs(dimension, 0);
    for(int i = 0; i < dimension; ++i) {
        for(int point = 0; point < points; ++point)
            std_devs[i] += pow(X[point][i] - avgs[i], 2);
        std_devs[i] = sqrtf(std_devs[i] / points);
    }
    
    return std_devs;
}

double LogisticRegression::standard_deviation(const std::vector<double>& x) {
    int points = x.size();

    double avg = mean(x);
    double std_dev = 0;
    for(int point = 0; point < points; ++point)
        std_dev += pow(x[point] - avg, 2);
    std_dev = sqrtf(std_dev / points);

    return std_dev;
}

void LogisticRegression::fit(const std::vector<std::vector<double>>& X) {
    int parameters = X[0].size() + 1;
    weights.reserve(parameters);

    for(int i = 0; i < parameters; ++i)
        weights.push_back(0);
}