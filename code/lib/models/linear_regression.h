#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "ml.h"

namespace Wally {
    class LinearRegression : public Regressor {
        virtual void compute_errors(const Vector<float>&);
        virtual void compute_gradient(const Matrix<float>&);
        virtual void gradient_descent(const float);
        virtual float mse_loss();

        virtual void fit(const Matrix<float>&) override;
        virtual void predict_(const Matrix<float>&);

    protected:
        struct LinearRegressionWorkbench {
            Vector<float> y_hat;
            Vector<float> errors;
            float grad_b;
            Vector<float> grad;

            LinearRegressionWorkbench(const std::size_t N, const std::size_t D)
                : y_hat(Vector<float>(N, 0.0f)), errors(Vector<float>(N, 0.0f)), grad_b(0.0f), grad(Vector<float>(D, 0.0f)) {}

            ~LinearRegressionWorkbench() = default;
        };

        float bias;
        Vector<float> weights;
        LinearRegressionWorkbench workbench;

    public:
        LinearRegression();

        virtual void train(const Matrix<float>&, const Vector<float>&, const float, const float, const std::size_t) override;
        virtual Vector<float> predict(const Matrix<float>&) override;
        virtual void params() override;

        ~LinearRegression() = default;
    };
};

#endif