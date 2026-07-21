#include "../lib/mlearn.h"

double Stats::mean(const vector<int>& x) {
    std::size_t points = x.size();

    int total = 0;
    for(std::size_t i = 0; i < points; ++i)
        total += x[i];

    return total / points;
}

double Stats::mean(const vector<double>& x) {
    std::size_t points = x.size();

    double total = 0;
    for(std::size_t i = 0; i < points; ++i)
        total += x[i];

    return total / points;
}

vector<double> Stats::mean(const matrix<int>& X) {
    std::size_t points = X.size();
    std::size_t features = X[0].size();

    vector<double> avgs(features, 0);
    for(std::size_t j = 0; j < features; ++j) {
        for(std::size_t i = 0; i < points; ++i)
            avgs[j] += X[i][j];
        avgs[j] /= points;
    }

    return avgs;
}

vector<double> Stats::mean(const matrix<double>& X) {
    std::size_t points = X.size();
    std::size_t features = X[0].size();

    vector<double> avgs(features, 0);
    for(std::size_t j = 0; j < features; ++j) {
        for(std::size_t i = 0; i < points; ++i)
            avgs[j] += X[i][j];
        avgs[j] /= points;
    }

    return avgs;
}

double Stats::var(const vector<int>& x) {
    std::size_t points = x.size();

    double avg = mean(x);
    double total = 0;
    for(std::size_t i = 0; i < points; ++i)
        total += pow(x[i] - avg, 2);
    
    return total / points;
}

double Stats::var(const vector<double>& x) {
    std::size_t points = x.size();

    double avg = mean(x);
    double total = 0;
    for(std::size_t i = 0; i < points; ++i)
        total += pow(x[i] - avg, 2);
    
    return total / points;
}

vector<double> Stats::var(const matrix<int>& X) {
    std::size_t points = X.size();
    std::size_t features = X[0].size();

    vector<double> avgs = mean(X);
    vector<double> vars(features, 0);
    for(std::size_t j = 0; j < features; ++j) {
        for(std::size_t i = 0; i < points; ++i)
            vars[j] += pow(X[i][j] - avgs[j], 2);
        vars[j] /= points;
    }
    
    return vars;
}

vector<double> Stats::var(const matrix<double>& X) {
    std::size_t points = X.size();
    std::size_t features = X[0].size();

    vector<double> avgs = mean(X);
    vector<double> vars(features, 0);
    for(std::size_t j = 0; j < features; ++j) {
        for(std::size_t i = 0; i < points; ++i)
            vars[j] += pow(X[i][j] - avgs[j], 2);
        vars[j] /= points;
    }
    
    return vars;
}

double Stats::stdd(const vector<int>& x) {
    return sqrtf(var(x));
}

double Stats::stdd(const vector<double>& x) {
    return sqrtf(var(x));
}

vector<double> Stats::stdd(const matrix<int>& X) {
    std::size_t features = X[0].size();

    vector<double> vars = var(X);
    for(std::size_t j = 0; j < features; ++j)
        vars[j] = sqrtf(vars[j]);

    return vars;
}

vector<double> Stats::stdd(const matrix<double>& X) {
    std::size_t features = X[0].size();

    vector<double> vars = var(X);
    for(std::size_t j = 0; j < features; ++j)
        vars[j] = sqrtf(vars[j]);

    return vars;
}

vector<double> Stats::norm(const vector<double>& x) {
    std::size_t points = x.size();

    double avg = mean(x);
    double std_dev = stdd(x);
    vector<double> norm(points, 0);
    for(std::size_t i = 0; i < points; ++i)
        norm[i] = (x[i] - avg) / std_dev;

    return norm;
}

matrix<double> Stats::norm(const matrix<double>& X) {
    std::size_t points = X.size();
    std::size_t features =  X[0].size();

    vector<double> avgs = mean(X);
    vector<double> std_devs = stdd(X);
    vector<std::vector<double>> norms(points, vector<double>(features, 0));
    for(std::size_t j = 0; j < features; ++j) {
        for(std::size_t i = 0; i < points; ++i)
            norms[i][j] = (X[i][j] - avgs[j]) / std_devs[j];
    }

    return norms;
}