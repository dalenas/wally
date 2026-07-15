#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "mlearn.h"

constexpr int BIAS = 0;

class LinearRegression {
private:
    std::vector<double> compute_gradient(const std::vector<double>&, const std::vector<std::vector<double>>&);
    void gradient_descent(const std::vector<double>&, const double&);
    std::vector<double> errors(const std::vector<double>&, const std::vector<double>&);
    double mean_squared_error(const std::vector<double>&);
    std::vector<double> mean(const std::vector<std::vector<double>>&);
    double mean(const std::vector<double>&);
    std::vector<double> standard_deviation(const std::vector<std::vector<double>>&);
    double standard_deviation(const std::vector<double>&);
protected:
    std::vector<double> weights;
public:
    LinearRegression() = default;
    void fit(const std::vector<std::vector<double>>&);
    void train(const std::vector<std::vector<double>>&, const std::vector<double>&, double, double, int);
    std::vector<double> predict(const std::vector<std::vector<double>>&);
    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>>&);
    std::vector<double> normalize(const std::vector<double>&);
    void _params();
    ~LinearRegression() = default;
};

#endif