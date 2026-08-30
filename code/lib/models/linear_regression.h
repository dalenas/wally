#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "ml.h"

namespace Wally {
    class LinearRegression : public Regressor {
        virtual Vector<float> compute_errors(const Vector<float>&, const Vector<float>&);
        virtual Vector<float> compute_gradient(const Vector<float>&, const Matrix<float>&);
        virtual void gradient_descent(const Vector<float>&, const float&);
        virtual double mse_loss(const Vector<float>&);

        virtual void fit(const Matrix<float>&) override;

    protected:
        float bias;
        Vector<float> weights;

    public:
        LinearRegression() = default;

        virtual void train(const Matrix<float>&, const Vector<float>&, const float, const float, const std::size_t) override;
        virtual Vector<float> predict(const Matrix<float>&) override;
        virtual void _params() override;

        ~LinearRegression() = default;
    };
}

#endif