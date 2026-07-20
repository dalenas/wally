#ifndef MLEARN_H
#define MLEARN_H

#include <cmath>
#include <iostream>
#include <vector>

using matrix = std::vector<std::vector<double>>;

class Regressor {
private:
    using targets = std::vector<double>;
public:
    virtual void fit(const matrix& X, const targets&, double, double, int) = 0;
};

class Classifier {
private:
    using labels = std::vector<int>;
public:
    virtual void fit(const matrix& X, const labels&, double, double, int) = 0;
};

#endif