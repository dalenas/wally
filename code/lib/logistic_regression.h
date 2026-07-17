#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "mlearn.h"

class LogisticRegression {
private:
    std::vector<std::vector<double>> weights;

    std::vector<double> mean(const std::vector<std::vector<double>>&);
    double mean(const std::vector<double>&);
    std::vector<double> standard_deviation(const std::vector<std::vector<double>>&);
    double standard_deviation(const std::vector<double>&);
    std::vector<std::vector<int>> one_hot_encoding(const std::vector<int>&);

    int class_count(const std::vector<int>&);
    double z(const std::vector<double>&, const int&);
    //std::vector<double> z(const std::vector<double>&);
    //std::vector<std::vector<double>> Z(const std::vector<std::vector<double>>&);
    double sigmoid_k(const std::vector<double>&, const int&);
    std::vector<double> sigmoid(const std::vector<double>&);
    std::vector<std::vector<double>> sigmoid(const std::vector<std::vector<double>>&);
    std::vector<double> softmax(const std::vector<double>&);
    std::vector<std::vector<double>> softmax(const std::vector<std::vector<double>>&);
    int softmax_max(const std::vector<double>&);

    std::vector<double> compute_errors(const std::vector<int>&, const std::vector<std::vector<double>>&);
    double logistic_loss(const std::vector<std::vector<double>>&, const std::vector<int>&);
    double cross_entropy(const std::vector<std::vector<double>>&, const std::vector<int>&);
    std::vector<double> compute_gradient(const std::vector<double>&, const std::vector<std::vector<double>>&);
    void gradient_descent(const std::vector<double>&, const double&);
public:
    LogisticRegression() = default;
    void fit(const std::vector<std::vector<double>>&, const std::vector<int>&);
    void train(const std::vector<std::vector<double>>&, const std::vector<int>&, double, double, int);
    std::vector<int> predict(const std::vector<std::vector<double>>&);
    ~LogisticRegression() = default;
};

#endif