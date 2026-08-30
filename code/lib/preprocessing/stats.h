#ifndef STATS_H
#define STATS_H

#include "../interface/containers.h"

namespace Wally::Stats {
    float mean(const Vector<float>&);
    Vector<float> mean(const Matrix<float>&);

    float var(const Vector<float>&);
    Vector<float> var(const Matrix<float>&);
    
    float stdd(const Vector<float>&);
    Vector<float> stdd(const Matrix<float>&);

    Vector<float> norm(const Vector<float>&);
    Matrix<float> norm(const Matrix<float>&);
}

#endif