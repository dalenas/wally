#ifndef CONTAINERS_H
#define CONTAINERS_H

#include <vector>

namespace Wally {
    enum class Major {
        row = 0,
        col = 1
    };

    template <typename T>
    class Vector {
        std::vector<T> data_;
        std::size_t size_;

    public:
        Vector(std::size_t);
        Vector(std::size_t, T);

        Vector(const Vector<T>&);
        Vector<T>& operator=(const Vector<T>&);

        std::size_t size() const;

        T* data();
        const T* data() const;

        T& operator()(std::size_t);
        const T& operator()(std::size_t) const;

        ~Vector() = default;
    };

    template <typename U>
    class Matrix {
        std::vector<U> data_;
        std::size_t rows_;
        std::size_t cols_;
        Major axis_;

    public:
        Matrix(std::size_t, std::size_t);
        Matrix(std::size_t, std::size_t, Major);
        Matrix(std::size_t, std::size_t, Major, U);

        Matrix(const Matrix<U>&);
        Matrix<U>& operator=(const Matrix<U>&);

        U& operator()(std::size_t, std::size_t);
        const U& operator()(std::size_t, std::size_t) const;

        std::size_t rows() const;
        std::size_t cols() const;
        std::size_t size() const;
        Major axis() const;
        
        U* data();
        const U* data() const;

        ~Matrix() = default;
    };
}

#include "interface/containers.tpp"

#endif