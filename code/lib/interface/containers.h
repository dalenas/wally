#ifndef CONTAINERS_H
#define CONTAINERS_H

#include <iostream>
#include <vector>

namespace Wally {
    enum class Major {
        row = 0,
        col = 1
    };

    template <typename T>
    class Vector {
        std::size_t size_;
        std::vector<T> data_;

    public:
        Vector(std::size_t);
        Vector(std::size_t, T);
        Vector(const std::vector<std::vector<T>>&);

        Vector(const Vector<T>&);
        Vector<T>& operator=(const Vector<T>&);

        std::size_t size() const;

        T* data();
        const T* data() const;

        T& operator()(std::size_t);
        const T& operator()(std::size_t) const;

        template<typename S>
        friend std::ostream& operator<<(std::ostream& out, const Vector<S>& v);

        ~Vector() = default;
    };

    template <typename U>
    class Matrix {
        std::size_t rows_;
        std::size_t cols_;
        Major axis_;
        std::vector<U> data_;

    public:
        Matrix(std::size_t, std::size_t);
        Matrix(std::size_t, std::size_t, Major);
        Matrix(std::size_t, std::size_t, Major, U);
        Matrix(const std::vector<std::vector<U>>&);

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

        template<typename V>
        friend std::ostream& operator<<(std::ostream& out, const Matrix<V>& M);

        ~Matrix() = default;
    };

    template<typename S>
    std::ostream& operator<<(std::ostream& out, const Vector<S>& v) {
        const std::size_t size = v.size();

        std::cout << "{";
        for(std::size_t i = 0; i < size-1; ++i)
            out << v(i) << ", ";
        out << v(size-1) << "}" << std::endl;

        return out;
    }

    template<typename V>
    std::ostream& operator<<(std::ostream& out, const Matrix<V>& M) {
        const std::size_t rows = M.rows();
        const std::size_t cols = M.cols();
        
        std::cout << "{";
        for(std::size_t i = 0; i < rows; ++i) {
            std::cout << "{";
            for(std::size_t j = 0; j < cols-1; ++j)
                out << M(i, j) << ", ";
            out << M(i, cols-1) << "}" << std::endl;
        }
        std::cout << "}" << std::endl;

        return out;
    }
};

#include "interface/containers.tpp"

#endif