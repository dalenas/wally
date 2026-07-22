#ifndef MULTI_LOGISTIC_REGRESSION_H
#define MULTI_LOGISTIC_REGRESSION_H

#include "mlearn.h"

class MultiLogisticRegression : public Classifier {
private:
    matrix<double> weights;

    double z(const vector<double>&, const int&);
    vector<double> softmax(const vector<double>&);
    matrix<double> softmax(const matrix<double>&);

    double cross_entropy_loss(const vector<int>&, const vector<double>&);
    vector<double> compute_errors(const vector<double>&, const vector<double>&);
    vector<double> compute_gradient(const matrix<double>&, const vector<double>&, const vector<double>&);
    void gradient_descent(const matrix<double>&, const vector<double>&, const vector<double>&, const double&);

    virtual void fit(const matrix<double>&) override;
public:
    MultiLogisticRegression() = default;
    virtual void train(const matrix<double>&, const vector<int>&, const double&, const double&, const std::size_t&) override;
    virtual vector<int> predict(const matrix<double>&) override;
    virtual void _params() override;
    ~MultiLogisticRegression() = default;
};

#endif