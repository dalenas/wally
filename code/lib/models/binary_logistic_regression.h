#ifndef BINARY_LOGISTIC_REGRESSION_H
#define BINARY_LOGISTIC_REGRESSION_H

#include "ml.h"

namespace Wally {
    class BinaryLogisticRegression : public Classifier {
        float bias;
        Vector<float> weights;

        float z(const Vector<float>&);
        float sigmoid(const Vector<float>&);
        Vector<float> sigmoid(const Matrix<float>&);

        Vector<float> compute_errors(const Vector<int>&, const Vector<float>&);
        float log_loss(const Vector<int>&, const Vector<float>&);
        Vector<float> compute_gradient(const Matrix<float>&, const Vector<int>&, const Vector<float>&);
        void gradient_descent(const Vector<float>&, const float);

        virtual void fit(const Matrix<float>&, const Vector<int>&) override;
        
    public:
        BinaryLogisticRegression() = default;
        virtual void train(const Matrix<float>&, const Vector<int>&, const float, const float, const std::size_t) override;
        virtual Vector<int> predict(const Matrix<float>&) override;
        virtual void _params() override;
        ~BinaryLogisticRegression() = default;
    };
}

#endif