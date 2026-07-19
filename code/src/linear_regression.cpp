#include "../lib/linear_regression.h"

std::vector<double> LinearRegression::mean(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int dimension = X[0].size();

    std::vector<double> avgs(dimension, 0);
    for(int i = 0; i < dimension; ++i) {
        for(int point = 0; point < points; ++point)
            avgs[i] += X[point][i];
        avgs[i] /= points;
    }

    return avgs;
}

double LinearRegression::mean(const std::vector<double>& x) {
    int points = x.size();

    double avg = 0;
    for(int point = 0; point < points; ++point)
        avg += x[point];
    
    avg /= points;
    return avg;
}

std::vector<double> LinearRegression::standard_deviation(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int dimension = X[0].size();

    std::vector<double> avgs = mean(X);
    std::vector<double> std_devs(dimension, 0);
    for(int i = 0; i < dimension; ++i) {
        for(int point = 0; point < points; ++point)
            std_devs[i] += pow(X[point][i] - avgs[i], 2);
        std_devs[i] = sqrtf(std_devs[i] / points);
    }
    
    return std_devs;
}

double LinearRegression::standard_deviation(const std::vector<double>& x) {
    int points = x.size();

    double avg = mean(x);
    double std_dev = 0;
    for(int point = 0; point < points; ++point)
        std_dev += pow(x[point] - avg, 2);
    std_dev = sqrtf(std_dev / points);

    return std_dev;
}

std::vector<double> LinearRegression::errors(const std::vector<double>& y_pred, const std::vector<double>& y) {
    int points = y.size();

    std::vector<double> errors(points, 0);
    for(int point = 0; point < points; ++point)
        errors[point] = y_pred[point] - y[point];
    
    return errors;
}

double LinearRegression::mean_squared_error(const std::vector<double>& errors) {
    int points = errors.size();

    double mse = 0;
    for(int point = 0; point < points; ++point)
        mse += pow(errors[point], 2);
    mse /= points;
    
    return mse;
}

std::vector<double> LinearRegression::compute_gradient(const std::vector<double>& errors, const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int parameters = weights.size();

    std::vector<double> gradient(parameters, 0);
    for(int point = 0; point < points; ++point)
        gradient[BIAS] += errors[point];
    gradient[BIAS] = 2 * gradient[BIAS] / points;
    for(int i = 1; i < parameters; ++i) {
        for(int point = 0; point < points; ++point)
            gradient[i] += errors[point]*X[point][i-1];
        gradient[i] = 2 * gradient[i] / points;
    }

    return gradient;
}

void LinearRegression::gradient_descent(const std::vector<double>& gradient, const double& learning_rate) {
    int parameters = weights.size();

    for(int i = 0; i < parameters; ++i)
        weights[i] -= learning_rate*gradient[i];
}

void LinearRegression::fit(const std::vector<std::vector<double>>& X) {
    int parameters = X[0].size() + 1;
    weights = std::vector<double>(parameters, 0);
}

void LinearRegression::train(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double learning_rate=0.05, double tol=1e-6, int max_iter=100) {
    for(int step = 0; step < max_iter; ++step) {
        std::vector<double> y_pred = predict(X);
        std::vector<double> y_error = errors(y_pred, y);

        if(mean_squared_error(y_error) < tol)
            return;
        
        std::vector<double> gradient = compute_gradient(y_error, X);
        gradient_descent(gradient, learning_rate);
    }
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int parameters = weights.size();
    std::vector<double> y_pred(points, weights[0]);

    for(int point = 0; point < points; ++point) {
        for(int i = 1; i < parameters; ++i)
            y_pred[point] += weights[i]*X[point][i-1];
    }

    return y_pred;
}

std::vector<std::vector<double>> LinearRegression::normalize(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int dimension =  X[0].size();

    std::vector<double> avgs = mean(X);
    std::vector<double> std_devs = standard_deviation(X);
    std::vector<std::vector<double>> X_norm(points, std::vector<double>(dimension, 0));
    for(int i = 0; i < dimension; ++i) {
        std::cout << "Feature #" << i+1 << ": Mean = " << avgs[i] << ", STD DEV = " << std_devs[i] << std::endl;
        for(int point = 0; point < points; ++point)
            X_norm[point][i] = (X[point][i] - avgs[i]) / std_devs[i];
    }

    return X_norm;
}

std::vector<double> LinearRegression::normalize(const std::vector<double>& x) {
    int points = x.size();

    double avg = mean(x);
    double std_dev = standard_deviation(x);
    std::vector<double> x_norm(points, 0);
    std::cout << "Target: Mean = " << avg << ", STD DEV = " << std_dev << std::endl;
    for(int point = 0; point < points; ++point)
        x_norm[point] = (x[point] - avg) / std_dev;

    return x_norm;
}

void LinearRegression::_params() {
    int params = weights.size()-1;

    std::cout << '[';
    for(int i = 0; i < params; ++i)
        std::cout << weights[i] << ", ";
    std::cout << weights[params] << ']';
}