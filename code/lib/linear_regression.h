#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <cmath>
#include <vector>

class LinearRegression {
private:
    std::vector<double> weights;

    std::vector<double> compute_gradient(std::vector<double>, std::vector<std::vector<double>>);
    std::vector<double> gradient_descent(std::vector<double>, std::vector<double>, double);
    double mean_squared_error(std::vector<double>);
    std::vector<double> mean(std::vector<std::vector<double>>);
    double mean(std::vector<double>);
    std::vector<double> standard_deviation(std::vector<std::vector<double>>);
    double standard_deviation(std::vector<double>);

public:
    LinearRegression() {};
    void fit(std::vector<std::vector<double>>, std::vector<double>);
    void train(std::vector<std::vector<double>>, std::vector<double>, double, double, int);
    std::vector<double> predict(std::vector<std::vector<double>>);
    void normalize(std::vector<std::vector<double>>);
    void normalize(std::vector<double>);
    ~LinearRegression();
};

#endif