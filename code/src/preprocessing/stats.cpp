#include "preprocessing/stats.h"

namespace Wally::Stats {
    Vector<float> norm(const Vector<float>& x) {
        const std::size_t N = x.size();
        const float mu = mean(x);
        const float sigma = stdd(x);

        Vector<float> z(N);
        Abstract::SIMD::sub(x, mu, z);
        Abstract::SIMD::div(z, sigma);

        return z;
    }

    Matrix<float> norm(const Matrix<float>& X) {
        const std::size_t N = X.rows();
        const std::size_t M = X.cols();
        const Vector<float> mus = mean(X);
        const Vector<float> sigmas = stdd(X);

        Matrix<float> Z_sigma(N, M, 0.0f);
        Abstract::SIMD::sub(X, mus, Z_sigma);

        Matrix<float> Z(N, M, 0.0f);              // Will make a simd compound div so i dont have to make another object
        Abstract::SIMD::div(Z_sigma, sigmas, Z);

        return Z;
    }
};