#ifndef BINARY_LOGISTIC_REGRESSION_H
#define BINARY_LOGISTIC_REGRESSION_H

#include "ml.h"

namespace Wally {
    class BinaryRegression : public Classifier {
        struct Workbench {
            Vector<float> logits;
            Vector<float> probs;
            
            Vector<int> y_hat;
            Vector<float> errors;
            
            float grad_b;
            Vector<float> grad;

            Workbench(const std::size_t N, const std::size_t D)
                : logits(Vector<float>(N, 0.0f)), probs(Vector<float>(N, 0.0f)), 
                y_hat(Vector<int>(N, 0)), errors(Vector<float>(N, 0.0f)), 
                grad_b(0.0f), grad(Vector<float>(D, 0.0f)) {}

            ~Workbench() = default;
        }

        float bias;
        Vector<float> weights;
        Workbench workbench;

        void z(const Matrix<float>&);
        void sigmoid();

        void compute_errors(const Vector<int>&);
        float log_loss(const Vector<int>&);
        void compute_gradient(const Matrix<float>&);
        void gradient_descent(const float);

        virtual void fit(const Matrix<float>&) override;
        virtual void predict_(const Matrix<float>&);
        
    public:
        BinaryRegression();
        virtual void train(const Matrix<float>&, const Vector<int>&, const float, const float, const std::size_t) override;
        virtual Vector<int> predict(const Matrix<float>&) override;
        virtual void params() override;
        ~BinaryRegression() = default;
    };
}

#endif