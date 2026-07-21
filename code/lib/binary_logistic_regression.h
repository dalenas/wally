#ifndef BINARY_LOGISTIC_REGRESSION_H
#define BINARY_LOGISTIC_REGRESSION_H

#include "mlearn.h"

class BinaryLogisticRegression : public Classifier {
private:
    vector<double> weights;

    double z(const vector<double>&);
    double sigmoid(const vector<double>&);
    vector<double> sigmoid(const matrix<double>&);

    vector<double> compute_errors(const vector<int>&, const vector<double>&);
    double log_loss(const vector<int>&, const vector<double>&);
    vector<double> compute_gradient(const matrix<double>&, const vector<int>&, const vector<double>&);
    void gradient_descent(const vector<double>&, const double&);

    virtual void fit(const matrix<double>&, const vector<int>&) override;
public:
    BinaryLogisticRegression() = default;
    virtual void train(const matrix<double>&, const vector<int>&, const double&, const double&, const std::size_t&) override;
    virtual vector<int> predict(const matrix<double>&) override;
    virtual void _params() override;
    ~BinaryLogisticRegression() = default;
};

#endif