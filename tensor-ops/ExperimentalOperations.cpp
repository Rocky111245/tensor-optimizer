//
// Created by Rocky170 on 11/27/2025.
// The purpose of this class is to experiment with tensors so that they can be optimized by comparing with Eigen.

#include "ExperimentalOperations.h"
#include <random>

ETensorWrapper::ETensorWrapper(int rows, int cols, int depth)
    : tensor_(rows, cols, depth) {
}

void ETensorWrapper::randomize(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (int d = 0; d < tensor_.depth(); ++d) {
        for (int r = 0; r < tensor_.rows(); ++r) {
            for (int c = 0; c < tensor_.columns(); ++c) {
                tensor_(r, c, d) = dist(gen);
            }
        }
    }
}
/* ==============================  Dimension Getters  ============================== */

int ETensor::rows() const noexcept {
    return rows_;
}

int ETensor::columns() const noexcept {
    return columns_;
}

int ETensor::depth() const noexcept {
    return depth_;
}

void ETensorWrapper::multiplyMatmulSpecial(const ETensorWrapper& other,
                                           ETensorWrapper& result) const {
    // Here we assume shapes are already correct:
    // this->tensor_ : M x K x D
    // other.tensor_: K x N x D
    // result.tensor_: M x N x D

    Tensor_Multiply_Tensor(result.tensor_, this->tensor_, other.tensor_);
}

void ETensorWrapper::multiplyMatmulSimd(const ETensorWrapper& other,
                                        ETensorWrapper& result) const {
    Tensor_Multiply_Tensor_SIMD(result.tensor_, this->tensor_, other.tensor_);
}
