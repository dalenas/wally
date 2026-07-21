#ifndef MLEARN_H
#define MLEARN_H

#include <cmath>
#include <iostream>
#include <vector>

using std::vector;
template <typename T>
using matrix = vector<vector<T>>;

namespace Stats {
    double mean(const vector<int>&);
    double mean(const vector<double>&);
    vector<double> mean(const matrix<int>&);
    vector<double> mean(const matrix<double>&);

    double var(const vector<int>&);
    double var(const vector<double>&);
    vector<double> var(const matrix<int>&);
    vector<double> var(const matrix<double>&);

    double stdd(const vector<int>&);
    double stdd(const vector<double>&);
    vector<double> stdd(const matrix<int>&);
    vector<double> stdd(const matrix<double>&);

    vector<double> norm(const vector<double>&);
    matrix<double> norm(const matrix<double>&);
};

class Regressor {
private:
    virtual void fit(const matrix<double>&) = 0;
public:
    Regressor() = default;
    virtual void train(const matrix<double>&, const vector<double>&, const double&, const double&, const std::size_t&) = 0;
    virtual vector<double> predict(const matrix<double>&) = 0;
    virtual void _params() = 0;
    ~Regressor() = default;
};

class Classifier {
private:
    virtual void fit(const matrix<double>&, const vector<int>&) = 0;
public:
    Classifier() = default;
    virtual void train(const matrix<double>&, const vector<int>&, const double&, const double&, const std::size_t&) = 0;
    virtual vector<int> predict(const matrix<double>&) = 0;
    virtual void _params() = 0;
    ~Classifier() = default;
};

#endif