#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "mlearn.h"

class Preprocessing {
private:

public:
    double mean(std::vector<double>);
    std::vector<double> mean(std::vector<std::vector<double>>);
    double var(std::vector<double>);
    std::vector<double> var(std::vector<std::vector<double>>);
    double stdd(std::vector<double>);
    std::vector<double> stdd(std::vector<std::vector<double>>);
    double cov(std::vector<double>, std::vector<double>);
    std::vector<double> cov(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
    double corr(std::vector<double>, std::vector<double>);
    std::vector<double> corr(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
    double norm(std::vector<double>);
    std::vector<double> norm(std::vector<std::vector<double>>);
};

#endif