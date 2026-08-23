#ifndef SIMD_H
#define SIMD_H

#include <immintrin.h>

#include "matrix.h"

namespace SIMD {
    namespace SIMD_ {
        __m256 load(const Matrix&);
        void store(Matrix&, __m256);
    }
    
    float sum(const Matrix&);
    float dot(const Matrix&);

    void add(const Matrix&, const Matrix&, Matrix&);
    void sub(const Matrix&, const Matrix&, Matrix&);
    void mult(float, const Matrix&, Matrix&);
    void mult(const Matrix&, const Matrix&, Matrix&);

    void mult_comp_add(Matrix&, const Matrix&, const Matrix&);
    void mult_comp_sub(Matrix&, float, const Matrix&);
}

#endif