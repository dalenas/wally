#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "mlearn.h"

class LinearRegression : public Regressor {
private:
    vector<double> compute_gradient(const vector<double>&, const matrix<double>&);
    void gradient_descent(const vector<double>&, const double&);
    vector<double> errors(const vector<double>&, const vector<double>&);
    double mse_loss(const vector<double>&);

    virtual void fit(const matrix<double>&) override;

protected:
    vector<double> weights;

public:
    LinearRegression() = default;
    virtual void train(const matrix<double>&, const vector<double>&, const double&, const double&, const std::size_t&) override;
    virtual vector<double> predict(const matrix<double>&) override;
    void _params();
    ~LinearRegression() = default;
};

#endif