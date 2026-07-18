#ifndef BINARY_LOGISTIC_REGRESSION_H
#define BINARY_LOGISTIC_REGRESSION_H

#include "mlearn.h"

class BinaryLogisticRegression {
private:
    std::vector<double> weights;

    double mean(const std::vector<double>&);
    std::vector<double> mean(const std::vector<std::vector<double>>&);
    double standard_deviation(const std::vector<double>&);
    std::vector<double> standard_deviation(const std::vector<std::vector<double>>&);

    double z(const std::vector<double>&);
    std::vector<double> z(const std::vector<std::vector<double>>&);
    double sigmoid(const std::vector<double>&);
    std::vector<double> sigmoid(const std::vector<std::vector<double>>&);

    std::vector<double> compute_errors(const std::vector<double>&, const std::vector<double>&);
    double log_loss(const std::vector<double>&, const std::vector<double>&);
    std::vector<double> compute_gradient(const std::vector<std::vector<double>>&, const std::vector<double>&, const std::vector<double>&);
    void gradient_descent(const std::vector<double>&, const double&);
public:
    BinaryLogisticRegression() = default;
    std::vector<double> normalize(const std::vector<double>&);
    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>>&);
    void fit(const std::vector<std::vector<double>>&);
    void train(const std::vector<std::vector<double>>&, const std::vector<double>&, double, double, int);
    std::vector<int> predict(const std::vector<std::vector<double>>&);
    void _params();
    ~BinaryLogisticRegression() = default;
};

#endif