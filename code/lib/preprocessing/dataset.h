#ifndef DATASET_H
#define DATASET_H

#include <fstream>
#include <iostream>
#include <sstream>

#include "abstract/traits.h"

namespace Wally {
    Matrix<std::string> read_csv(const std::string&);

    Matrix<float> cvt_float(const Matrix<std::string>&, const std::size_t);
    Matrix<float> cvt_float(const Matrix<std::string>&, const std::vector<std::size_t>&);

    Matrix<float> get_features(const Matrix<float>&, const std::vector<std::size_t>&);
    Vector<float> get_target(const Matrix<float>&, const std::size_t);
};

#endif