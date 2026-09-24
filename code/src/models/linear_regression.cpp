#include "models/linear_regression.h"

namespace Wally {
    void LinearRegression::compute_errors(const Vector<float>& y) {
        Abstract::SIMD::sub(y, workbench.y_hat, workbench.errors);
    }

    void LinearRegression::compute_gradient(const Matrix<float>& X) {
        const float N = static_cast<float>(workbench.errors.size());
        
        if(X.axis() == Major::row)
            Abstract::SIMD::cross(workbench.errors, X, workbench.grad);
        else
            Abstract::SIMD::cross(X, workbench.errors, workbench.grad);
        Abstract::SIMD::div(workbench.grad, N);

        workbench.grad_b = Abstract::SIMD::sum(workbench.errors) / N;
    }

    void LinearRegression::gradient_descent(const float alpha) {
        bias -= alpha * workbench.grad_b;
        Abstract::SIMD::fmsub(weights, alpha, workbench.grad);
    }

    float LinearRegression::mse_loss() {
        const std::size_t N = workbench.errors.size();
        const float N_ps = static_cast<float>(N);

        return Abstract::SIMD::sqsum(workbench.errors) / N_ps;
    }

    void LinearRegression::fit(const Matrix<float>& X) {
        bias = 0.0f;
        weights = Vector<float>(X.cols(), 0.0f);
        workbench = LinearRegressionWorkbench(X.rows(), X.cols());
    }

    void LinearRegression::predict_(const Matrix<float>& X) {
        if(X.axis() == Major::row)
            Abstract::SIMD::cross(X, weights, workbench.y_hat);
        else
            Abstract::SIMD::cross(weights, X, workbench.y_hat);

        Abstract::SIMD::add(workbench.y_hat, bias);
        params();
    }

    LinearRegression::LinearRegression()
        : weights(0), workbench(0, 0) {}

    void LinearRegression::train(const Matrix<float>& X, const Vector<float>& y, const float alpha, const float tol, const std::size_t max_iter) {
        fit(X);
        for(std::size_t iter = 0; iter < max_iter; ++iter) {
            predict_(X);
            compute_errors(y);

            if(mse_loss() < tol)
                return;

            compute_gradient(X);
            gradient_descent(alpha);
        }
    }

    Vector<float> LinearRegression::predict(const Matrix<float>& X) {
        Vector<float> y_hat(X.rows());
        if(X.axis() == Major::row)
            Abstract::SIMD::cross(X, weights, y_hat);
        else
            Abstract::SIMD::cross(weights, X, y_hat);

        Abstract::SIMD::add(y_hat, bias);

        return y_hat;
    }

    void LinearRegression::params() {
        std::cout << "Bias = " << bias << std::endl;
        std::cout << "Weights = " << weights << std::endl;
    }
}