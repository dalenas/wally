#ifndef ML_H
#define ML_H

#include "../interface/containers.h"

constexpr std::size_t BIAS = 0;

namespace Wally {
    class Regressor {
        virtual void fit(const Matrix<float>&) = 0;

    public:
        Regressor() = default;

        virtual Vector<float> predict(const Matrix<float>&) = 0;
        virtual void train(const Matrix<float>&, const Vector<float>&, const float, const float, const std::size_t) = 0;
        virtual void _params() = 0;

        ~Regressor() = default;
    };

    class Classifier {
        virtual void fit(const Matrix<float>&, const Vector<int>&) = 0;

    public:
        Classifier() = default;

        virtual Vector<int> predict(const Matrix<float>&) = 0;
        virtual void train(const Matrix<float>&, const Vector<int>&, const float, const float, const std::size_t) = 0;
        virtual void _params() = 0;

        ~Classifier() = default;
    };
}

#endif