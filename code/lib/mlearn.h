#ifndef MLEARN_H
#define MLEARN_H

#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

constexpr std::size_t BIAS = 0;

using Eigen::VectorXi;
using Eigen::VectorXf;
using Eigen::MatrixXi;
using Eigen::MatrixXf;
using std::vector;
template <typename T>
using matrix = vector<vector<T>>;

namespace Stats {
    float mean(const VectorXi&);
    float mean(const VectorXf&);
    VectorXf mean(const MatrixXi&);
    VectorXf mean(const MatrixXf&);

    float var(const VectorXi&);
    float var(const VectorXf&);
    VectorXf var(const MatrixXi&);
    VectorXf var(const MatrixXf&);

    float stdd(const VectorXi&);
    float stdd(const VectorXf&);
    VectorXf stdd(const MatrixXi&);
    VectorXf stdd(const MatrixXf&);

    VectorXf norm(const VectorXf&);
    MatrixXf norm(const MatrixXf&);
};

class Regressor {
private:
    virtual void fit(const MatrixXf&) = 0;
public:
    Regressor() = default;
    virtual void train(const MatrixXf&, const VectorXf&, const double&, const double&, const std::size_t&) = 0;
    virtual VectorXf predict(const MatrixXf&) = 0;
    virtual void _params() = 0;
    ~Regressor() = default;
};

class Classifier {
private:
    virtual void fit(const MatrixXf&, const VectorXi&) = 0;
public:
    Classifier() = default;
    virtual void train(const MatrixXf&, const VectorXi&, const double&, const double&, const std::size_t&) = 0;
    virtual VectorXi predict(const MatrixXf&) = 0;
    virtual void _params() = 0;
    ~Classifier() = default;
};

#endif