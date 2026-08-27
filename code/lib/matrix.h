#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

using std::size_t;
using std::vector;

class Matrix {
    vector<float> data_;
    size_t rows_;
    size_t cols_;

public:
    Matrix(size_t, size_t);
    Matrix(size_t, size_t, float);

    float& operator()(size_t, size_t);
    const float& operator()(size_t, size_t) const;

    size_t rows() const;
    size_t cols() const;
    size_t size() const;
    
    float* data();
    const float* data() const;

    ~Matrix() = default;
};

#endif