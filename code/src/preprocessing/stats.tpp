#ifndef STATS_TPP
#define STATS_TPP

namespace Wally::Stats {
    template<typename T>
    float mean(const Vector<T>& x) {
        const std::size_t N = x.size();
        return SIMD::sum(x) / N;
    }

    template<typename T>
    Vector<float> mean(const Matrix<T>& X) {
        const std::size_t N = X.rows();
        const std::size_t M = X.cols();

        Vector<float> sums(M);
        SIMD::sum(X, sums);

        return SIMD::div(sums, N);
    }

    template<typename T>
    float var(const Vector<T>& x) {
        const std::size_t N = x.size();

        const float mu = mean(x);
        Vector<float> z(N);
        SIMD::sub(x, mu, z);

        return SIMD::sqsum(z) / N;
    }

    template<typename T>
    Vector<float> var(const Matrix<T>& X) {     // Want to make a simd overload for Matrix - Vector so I can just subtract a the means like that
        const std::size_t N = X.rows();         // Nevermind I just made it
        const std::size_t M = X.cols();

        Vector<float> mus = mean(X);
        Matrix<float> Z(N, M);
        SIMD::sub(X, mus, Z);

        Vector<float> sqsums(M);
        SIMD::sqsum(Z, sqsums);

        return SIMD::div(sqsums, N);
    }
    
    template<typename T>
    float stdd(const Vector<T>& x) {
        return sqrt(var(x));
    }

    template<typename T>
    Vector<float> stdd(const Matrix<T>& X) {
        const std::size_t M = X.cols();

        Vector<float> sigma_sqs = var(X);
        Vector<float> sigmas(M);
        SIMD::sqrt(sigma_sqs, sigmas);

        return sigmas;
    }
};

#endif