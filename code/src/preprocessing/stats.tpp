#ifndef STATS_TPP
#define STATS_TPP

namespace Wally::Stats {
    template<typename T>
    float mean(const Vector<T>& x) {
        const float N_ps = static_cast<float>(x.size());
        return Abstract::SIMD::sum(x) / N_ps;
    }

    template<typename T>
    Vector<float> mean(const Matrix<T>& X) {
        const std::size_t N = X.rows();
        const float N_ps = static_cast<float>(N);
        const std::size_t M = X.cols();

        Vector<float> mus(M);
        Abstract::SIMD::sum(X, mus);
        Abstract::SIMD::div(mus, N_ps);

        return mus;
    }

    template<typename T>
    float var(const Vector<T>& x) {
        const std::size_t N = x.size();
        const float N_ps = static_cast<float>(N);

        const float mu = mean(x);
        Vector<float> z(N);
        Abstract::SIMD::sub(x, mu, z);

        return Abstract::SIMD::sqsum(z) / N_ps;
    }

    template<typename T>
    Vector<float> var(const Matrix<T>& X) {     // Want to make a simd overload for Matrix - Vector so I can just subtract a the means like that
        const std::size_t N = X.rows();         // Nevermind I just made it
        const float N_ps = static_cast<float>(N);
        const std::size_t M = X.cols();

        Vector<float> mus = mean(X);
        Matrix<float> Z(N, M);
        Abstract::SIMD::sub(X, mus, Z);

        Vector<float> sigma_sqs(M);
        Abstract::SIMD::sqsum(Z, sigma_sqs);
        Abstract::SIMD::div(sigma_sqs, N_ps);

        return sigma_sqs;
    }
    
    template<typename T>
    float stdd(const Vector<T>& x) {
        float sigma = sqrt(var(x));
        // std::cout << "Stdd: " << sigma << std::endl;
        return sigma;
    }

    template<typename T>
    Vector<float> stdd(const Matrix<T>& X) {
        const std::size_t M = X.cols();

        Vector<float> sigma_sqs = var(X);
        Vector<float> sigmas(M);
        Abstract::SIMD::sqrt(sigma_sqs, sigmas);

        //std::cout << "Stdds: ";
        //std::cout << sigmas << std::endl;
        
        return sigmas;
    }
};

#endif