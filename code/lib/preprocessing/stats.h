#ifndef STATS_H
#define STATS_H

#include <cmath>

#include "abstract/simd.h"

namespace Wally::Stats {
    template<typename T>
    float mean(const Vector<T>&);
    template<typename T>
    Vector<float> mean(const Matrix<T>&);

    template<typename T>
    float var(const Vector<T>&);
    template<typename T>
    Vector<float> var(const Matrix<T>&);
    
    template<typename T>
    float stdd(const Vector<T>&);
    template<typename T>
    Vector<float> stdd(const Matrix<T>&);

    Vector<float> norm(const Vector<float>&);
    Matrix<float> norm(const Matrix<float>&);
}

#include "preprocessing/stats.tpp"

#endif