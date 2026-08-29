#ifndef CONTAINERS_H
#define CONTAINERS_H

#include <vector>

namespace Wally {
    template <typename T>
    class Vector {
        std::vector<T> data_;
        std::size_t size_;

    public:
        Vector(std::size_t);
        Vector(std::size_t, T);

        T& operator()(std::size_t);
        const T& operator()(std::size_t) const;

        const std::size_t size() const;

        T* data();
        const T* data() const;

        ~Vector() = default;
    };

    template <typename U>
    class Matrix {
        std::vector<U> data_;
        std::size_t rows_;
        std::size_t cols_;

    public:
        Matrix(std::size_t, std::size_t);
        Matrix(std::size_t, std::size_t, U);

        U& operator()(std::size_t, std::size_t);
        const U& operator()(std::size_t, std::size_t) const;

        const std::size_t rows() const;
        const std::size_t cols() const;
        const std::size_t size() const;
        
        U* data();
        const U* data() const;

        ~Matrix() = default;
    };
}

#endif