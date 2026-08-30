#ifndef MULTI_LOGISTIC_REGRESSION_H
#define MULTI_LOGISTIC_REGRESSION_H

#include "ml.h"

namespace Wally {
    class MultiLogisticRegression : public Classifier {
        Vector<float> biases;
        Matrix<float> weights;

        float z(const Vector<double>&, const int);
        Vector<float> softmax(const Vector<float>&);
        Matrix<float> softmax(const Matrix<float>&);

        double cross_entropy_loss(const Vector<int>&, const Matrix<float>&);
        Vector<float> compute_errors(const int, const Vector<float>&);
        Matrix<float> compute_errors(const Vector<int>&, const Matrix<float>&);
        Matrix<float> compute_gradient(const Matrix<float>&, const Vector<int>&, const Matrix<float>&);
        void gradient_descent(const Matrix<float>&, const Vector<int>&, const Matrix<float>&, const float);

        virtual void fit(const Matrix<float>&, const Vector<int>&) override;

    public:
        MultiLogisticRegression() = default;
        virtual void train(const Matrix<float>&, const Vector<int>&, const float, const float, const std::size_t) override;
        virtual Vector<int> predict(const Matrix<float>&) override;
        virtual void _params() override;
        ~MultiLogisticRegression() = default;
    };
}

#endif