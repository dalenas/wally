#ifndef MLEARN_H
#define MLEARN_H

#include <cmath>
#include <iostream>
#include <vector>

using matrix = std::vector<std::vector<double>>;

class Regressor {
private:
    using targets = std::vector<double>;
    virtual void fit(const matrix&) = 0;
public:
    Regressor() = default;
    virtual void train(const matrix&, const targets&, const double&, const double&, const std::size_t&) = 0;
    virtual targets predict(const matrix&) = 0;
    ~Regressor() = default;
};

class Classifier {
private:
    using labels = std::vector<int>;
    virtual void fit(const matrix&, const labels&) = 0;
public:
    Classifier() = default;
    virtual void train(const matrix& X, const labels&, const double&, const double&, const std::size_t&) = 0;
    virtual labels predict(const matrix&) = 0;
    ~Classifier() = default;
};

#endif