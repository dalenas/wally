#include "../../lib/interface/containers.h"

template<typename T>
Wally::Vector<T>::Vector(std::size_t s)
    : size_(s) { data_.reserve(size_); }

template<typename T>
Wally::Vector<T>::Vector(std::size_t s, T init)
    : size_(s) { data_.assign(size_, init); }

template<typename T>
T& Wally::Vector<T>::operator()(std::size_t i) {
    return data_[i];
}

template<typename T>
const T& Wally::Vector<T>::operator()(std::size_t) const {
    return data_[i];
}

template<typename T>
const std::size_t Wally::Vector<T>::size() const {
    return size_;
}

template<typename T>
T* Wally::Vector<T>::data() {
    return data_.data();
}

template<typename T>
const T* Wally::Vector<T>::data() const {
    return data_.data();
}

template<typename U>
Wally::Matrix<U>::Matrix(std::size_t r, std::size_t c, Major a)
    : rows_(r), cols_(c), axis_(a) { data_.reserve(rows_ * cols_); }

template<typename U>
Wally::Matrix<U>::Matrix(std::size_t r, std::size_t c, Major a, U init)
    : rows_(r), cols_(c), axis_(a) { data_.assign(rows_* cols_, init); }

template<typename U>
U& Wally::Matrix<U>::operator()(std::size_t row, std::size_t col) {
    return data_[col + row*cols_];
}

template<typename U>
const U& Wally::Matrix<U>::operator()(std::size_t row, std::size_t col) const {
    return data_[col + row*cols_];
}

template<typename U>
const std::size_t Wally::Matrix<U>::rows() const {
    return rows_;
}

template<typename U>
const std::size_t Wally::Matrix<U>::cols() const {
    return cols_;
}

template<typename U>
const std::size_t Wally::Matrix<U>::size() const {
    return rows_ * cols_;
}

template<typename U>
const std::size_t Wally::Matrix<U>::axis() const {
    return axis_;
}


template<typename U>
U* Wally::Matrix<U>::data() {
    return data_.data();
}

template<typename U>
const U* Wally::Matrix<U>::data() const {
    return data_.data();
}