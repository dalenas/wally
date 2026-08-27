#include "../lib/matrix.h"

Matrix::Matrix(size_t r, size_t c)
    : rows_(r), cols_(c) { data_.reserve(rows_ * cols_); }

Matrix::Matrix(size_t r, size_t c, float init)
    : rows_(r), cols_(c) { data_.assign(rows_* cols_, init); }

float& Matrix::operator()(size_t row, size_t col) {
    return data_[col + row*cols_];
}

const float& Matrix::operator()(size_t row, size_t col) const {
    return data_[col + row*cols_];
}

size_t Matrix::rows() const {
    return rows_;
}

size_t Matrix::cols() const {
    return cols_;
}

size_t Matrix::size() const {
    return rows_ * cols_;
}

float* Matrix::data() {
    return data_.data();
}

const float* Matrix::data() const {
    return data_.data();
}