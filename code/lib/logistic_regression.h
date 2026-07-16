#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "mlearn.h"

class LogisticRegression {
private:
    std::vector<double> weights;

    std::vector<double> compute_gradient(const std::vector<double>&, const std::vector<std::vector<double>>&);
    void gradient_descent(const std::vector<double>&, const double&);
    std::vector<double> mean(const std::vector<std::vector<double>>&);
    double mean(const std::vector<double>&);
    std::vector<double> standard_deviation(const std::vector<std::vector<double>>&);
    double standard_deviation(const std::vector<double>&);
    double sigmoid(std::vector<double>);
public:
    LogisticRegression() = default;
    void fit(const std::vector<std::vector<double>>&);
    void train(const std::vector<std::vector<double>>&, const std::vector<int>&, double, double, int);
    std::vector<double> predict(const std::vector<std::vector<double>>&);
    ~LogisticRegression() = default;
};

#endif