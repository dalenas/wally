#include "../lib/binary_logistic_regression.h"

double BinaryLogisticRegression::mean(const std::vector<double>& x) {
    int points = x.size();

    double avg = 0;
    for(int i = 0; i < points; ++i)
        avg += x[i];
    avg /= points;

    return avg;
}

std::vector<double> BinaryLogisticRegression::mean(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int features = X[0].size();

    std::vector<double> avgs(features, 0);
    for(int j = 0; j < features; ++j) {
        for(int i = 0; i < points; ++i)
            avgs[j] += X[i][j];
        avgs[j] /= points;
    }

    return avgs;
}

double BinaryLogisticRegression::standard_deviation(const std::vector<double>& x) {
    int points = x.size();

    double avg = mean(x);
    double std_dev = 0;
    for(int i = 0; i < points; ++i)
        std_dev += pow(x[i] - avg, 2);
    std_dev = sqrtf(std_dev / points);

    return std_dev;
}

std::vector<double> BinaryLogisticRegression::standard_deviation(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int features = X[0].size();

    std::vector<double> avgs = mean(X);
    std::vector<double> std_devs(features, 0);
    for(int j = 0; j < features; ++j) {
        for(int i = 0; i < points; ++i)
            std_devs[j] += pow(X[i][j] - avgs[j], 2);
        std_devs[j] = sqrtf(std_devs[j] / points);
    }
    
    return std_devs;
}

double BinaryLogisticRegression::z(const std::vector<double>& x) {
    int parameters = weights.size();

    double z = weights[0];
    for(int j = 1; j < parameters; ++j)
        z += weights[j]*x[j-1];

    return z;
}

std::vector<double> BinaryLogisticRegression::z(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int parameters = weights.size();
    
    std::vector<double> z(points, 0);
    for(int i = 0; i < points; ++i)
        z[i] = this->z(X[i]);

    return z;
}

double BinaryLogisticRegression::sigmoid(const std::vector<double>& x) {
    return 1 / (1 + exp(-z(x)));
}

std::vector<double> BinaryLogisticRegression::sigmoid(const std::vector<std::vector<double>>& X) {
    int points = X.size();

    std::vector<double> sigmoid(points, 0);
    for(int i = 0; i < points; ++i)
        sigmoid[i] = this->sigmoid(X[i]);

    return sigmoid;
}

std::vector<double> BinaryLogisticRegression::compute_errors(const std::vector<int>& y, const std::vector<double>& p) {
    int points = y.size();

    std::vector<double> errors(points, 0);
    for(int i = 0; i < points; ++i)
        errors[i] = p[i] - y[i];

    return errors;
}

double BinaryLogisticRegression::log_loss(const std::vector<int>& y, const std::vector<double>& p) {
    int points = y.size();

    double loss = 0;
    for(int i = 0; i < points; ++i)
        loss += -y[i]*log(p[i]) - (1 - y[i])*log(1 - p[i]);
    
    return loss;
}

std::vector<double> BinaryLogisticRegression::compute_gradient(const std::vector<std::vector<double>>& X, const std::vector<int>& y, const std::vector<double>& p) {
    int points = X.size();
    int parameters = weights.size();

    std::vector<double> errors = compute_errors(y, p);
    std::vector<double> gradient(parameters, 0);
    for(int i = 0; i < points; ++i)
        gradient[0] += errors[i];
    gradient[0] /= points;
    for(int j = 1; j < parameters; ++j) {
        for(int i = 0; i < points; ++i)
            gradient[j] += errors[i]*X[i][j-1];
        gradient[j] /= points;
    }

    return gradient;
}

void BinaryLogisticRegression::gradient_descent(const std::vector<double>& gradient, const double& learning_rate) {
    int parameters = gradient.size();

    for(int j = 0; j < parameters; ++j)
        weights[j] -= learning_rate*gradient[j];
}

std::vector<double> BinaryLogisticRegression::normalize(const std::vector<double>& x) {
    int points = x.size();

    double avg = mean(x);
    double std_dev = standard_deviation(x);
    std::vector<double> x_norm(points, 0);
    for(int i = 0; i < points; ++i)
        x_norm[i] = (x[i] - avg) / std_dev;

    return x_norm;
}

std::vector<std::vector<double>> BinaryLogisticRegression::normalize(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int features =  X[0].size();

    std::vector<double> avgs = mean(X);
    std::vector<double> std_devs = standard_deviation(X);
    std::vector<std::vector<double>> X_norm(points, std::vector<double>(features, 0));
    for(int j = 0; j < features; ++j) {
        for(int i = 0; i < points; ++i)
            X_norm[i][j] = (X[i][j] - avgs[j]) / std_devs[j];
    }

    return X_norm;
}

void BinaryLogisticRegression::fit(const std::vector<std::vector<double>>& X) {
    int parameters = X[0].size() + 1;
    weights = std::vector<double>(parameters, 0);
}

std::vector<int> BinaryLogisticRegression::predict(const std::vector<std::vector<double>>& X) {
    int points = X.size();

    std::vector<double> p = sigmoid(X);
    std::vector<int> y_pred(points, 0);
    for(int i = 0; i < points; ++i)
        if(p[i] > 0.5) y_pred[i] = 1;
    
    return y_pred;
}

void BinaryLogisticRegression::train(const std::vector<std::vector<double>>& X, const std::vector<int>& y, double learning_rate, double tol, int max_iter) {
    for(int k = 0; k < max_iter; ++k) {
        std::vector<double> p = sigmoid(X);

        if(abs(log_loss(y, p)) < tol)
            return;

        std::vector<double> gradient = compute_gradient(X, y, p);
        gradient_descent(gradient, learning_rate);
    }
}

void BinaryLogisticRegression::_params() {
    int params = weights.size() - 1;

    std::cout << '[';
    for(int i = 0; i < params; ++i)
        std::cout << weights[i] << ", ";
    std::cout << weights[params] << ']';
}