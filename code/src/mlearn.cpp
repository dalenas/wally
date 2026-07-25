#include "../lib/mlearn.h"

float Stats::mean(const VectorXi& x) {
    return x.mean();
}

float Stats::mean(const VectorXf& x) {
    return x.mean();
}

VectorXf Stats::mean(const MatrixXi& X) {
    return X.colwise().mean();
}

VectorXf Stats::mean(const MatrixXf& X) {
    return X.colwise().mean();
}

float Stats::var(const VectorXi& x) {
    return (x.array() - x.mean()).square().sum() / x.size();
}

float Stats::var(const VectorXf& x) {
    return (x.array() - x.mean()).square().sum() / x.size();
}

VectorXf Stats::var(const MatrixXi& X) {
    VectorXf vars(X.cols());
    for(std::size_t i = 0; i < X.cols(); ++i) {
        VectorXi x = X.col(i);
        vars(i) = var(x);
    }
    return vars;
}

VectorXf Stats::var(const MatrixXf& X) {
    VectorXf vars(X.cols());
    for(std::size_t i = 0; i < X.cols(); ++i) {
        VectorXf x = X.col(i);
        vars(i) = var(x);
    }
    return vars;
}

float Stats::stdd(const VectorXi& x) {
    return sqrtf(var(x));
}

float Stats::stdd(const VectorXf& x) {
    return sqrtf(var(x));
}

VectorXf Stats::stdd(const MatrixXi& X) {
    return var(X).array().sqrt().matrix();
}

VectorXf Stats::stdd(const MatrixXf& X) {
    return var(X).array().sqrt().matrix();
}

VectorXf Stats::norm(const VectorXf& x) {
    return ((x.array() - x.mean()) / stdd(x)).matrix();
}

MatrixXf Stats::norm(const MatrixXf& X) {
    MatrixXf norms(X.rows(), X.cols());
    for(std::size_t i = 0; i < X.cols(); ++i) {
        VectorXf x = X.col(i);
        norms.col(i) = norm(x);
    }
    return norms;
}