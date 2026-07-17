#include "../lib/logistic_regression.h"

std::vector<double> LogisticRegression::mean(const std::vector<std::vector<double>>& X) {
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

double LogisticRegression::mean(const std::vector<double>& x) {
    int points = x.size();

    double avg = 0;
    for(int point = 0; point < points; ++point)
        avg += x[point];
    
    avg /= points;
    return avg;
}

std::vector<double> LogisticRegression::standard_deviation(const std::vector<std::vector<double>>& X) {
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

double LogisticRegression::standard_deviation(const std::vector<double>& x) {
    int points = x.size();

    double avg = mean(x);
    double std_dev = 0;
    for(int point = 0; point < points; ++point)
        std_dev += pow(x[point] - avg, 2);
    std_dev = sqrtf(std_dev / points);

    return std_dev;
}

int LogisticRegression::class_count(const std::vector<int>& y) {
    int points = y.size();

    int classes = 0;
    for(int i = 0; i < points; ++i)
        if(y[i] >= classes) ++classes;

    return classes;
}

std::vector<std::vector<int>> LogisticRegression::one_hot_encoding(const std::vector<int>& y) {
    int points = y.size();
    int classes = weights[0].size();

    std::vector<std::vector<int>> one_hot(points, std::vector<int>(classes, 0));
    for(int point = 0; point < points; ++point)
        one_hot[point][y[point]] = 1;

    return one_hot;
}

double LogisticRegression::cross_entropy(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    int points = y.size();
    
    std::vector<std::vector<int>> one_hot = one_hot_encoding(y);

}

double LogisticRegression::z(const std::vector<double>& x, const int& k) {
    int parameters = weights[k].size();

    double z = weights[k][0];
    for(int i = 1; i < parameters; ++i)
        z += weights[k][i]*x[i-1];

    return z;
}

/*std::vector<double> LogisticRegression::z(const std::vector<double>& x) {
    int classes = weights.size();
    int parameters = weights[0].size();
    
    std::vector<double> z(parameters, 0);
    for(int k = 0; k < classes; ++k)
        z[k] = z_k(x, k);

    return z;
}*/

/*std::vector<std::vector<double>> LogisticRegression::Z(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int classes = weights.size();

    std::vector<std::vector<double>> Z(points, std::vector<double>(classes, 0));
    for(int point = 0; point < points; ++point)
        Z[point] = z(X[point]);
    
    return Z;
}*/

std::vector<double> LogisticRegression::softmax(const std::vector<double>& x) {
    int classes = weights.size();
    int parameters = weights[0].size();

    std::vector<double> exps(classes, 0);
    double norm = 0;
    for(int k = 0; k < classes; ++k) {
        exps[k] = exp(z(x, k));
        norm += exps[k];
    }

    std::vector<double> softmax(classes, 0);
    for(int k = 0; k < classes; ++k)
        softmax[k] = exps[k] / norm;

    return softmax;
}

std::vector<std::vector<double>> LogisticRegression::softmax(const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int classes = weights.size();

    std::vector<std::vector<double>> softmax_X(points, std::vector<double>(classes, 0));
    for(int point = 0; point < points; ++point)
        softmax_X[point] = softmax(X[point]);

    return softmax_X;
}

int LogisticRegression::softmax_max(const std::vector<double>& softmax) {
    int classes = weights.size();

    int max_index = 0;
    for(int k = 0; k < classes; ++k)
        if(softmax[k] > softmax[max_index]) max_index = k;
    
    return max_index;
}

std::vector<double> LogisticRegression::compute_errors(const std::vector<int>& y, const std::vector<std::vector<double>>& softmax) {
    int points = y.size();

    std::vector<double> errors(points, 0);
    for(int point = 0; point < points; ++point)
        errors[point] = 1 - softmax[point][y[point]];

    return errors;
}

std::vector<double> LogisticRegression::compute_gradient(const std::vector<double>& errors, const std::vector<std::vector<double>>& X) {
    int points = X.size();
    int parameters = weights.size();

    std::vector<double> gradient(parameters, 0);
    for(int point = 0; point < points; ++point)
        gradient[0] += errors[point];
    gradient[0] = -gradient[0] / points;
    for(int i = 1; i < parameters; ++i) {
        for(int point = 0; point < points; ++point)
            gradient[i] += errors[point]*X[point][i];
        gradient[i] = -gradient[i] / points;
    }

    return gradient;
}

void LogisticRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    int classes = class_count(y);
    int parameters = X.size() + 1;

    weights = std::vector<std::vector<double>>(classes, std::vector<double>(parameters, 0));
}

std::vector<int> LogisticRegression::predict(const std::vector<std::vector<double>>& X) {
    int points = X.size();

    std::vector<std::vector<double>> softmax_X = softmax(X);
    std::vector<int> y_pred(points, 0);
    for(int point = 0; point < points; ++point)
        y_pred[point] = softmax_max(softmax_X[point]);

    return y_pred;
}