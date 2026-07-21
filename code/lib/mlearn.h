#ifndef MLEARN_H
#define MLEARN_H

#include <cmath>
#include <iostream>
#include <vector>

using matrix = std::vector<std::vector<double>>;

class Regressor {
private:
    virtual void fit(const matrix&) = 0;
public:
    Regressor() = default;
    virtual void train(const matrix&, const std::vector<double>&, const double&, const double&, const std::size_t&) = 0;
    virtual std::vector<double> predict(const matrix&) = 0;
    ~Regressor() = default;
};

class Classifier {
private:
    virtual void fit(const matrix&, const std::vector<double>&) = 0;
public:
    Classifier() = default;
    virtual void train(const matrix& X, const std::vector<double>&, const double&, const double&, const std::size_t&) = 0;
    virtual std::vector<double> predict(const matrix&) = 0;
    ~Classifier() = default;
};

#endif