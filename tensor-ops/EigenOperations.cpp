//
// Created by Rocky170 on 11/25/2025.
//

#include "EigenOperations.h"
#include <random>

EigenTensor::EigenTensor(int rows, int cols, int depth)
    : rows_(rows), cols_(cols), depth_(depth), data_(rows, cols, depth) {
}

void EigenTensor::randomize(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            for (int d = 0; d < depth_; ++d) {
                data_(r, c, d) = dist(gen);
            }
        }
    }
}

void EigenTensor::multiply(const EigenTensor& other, EigenTensor& result) const {
    // Element-wise multiplication
    result.data_ = data_ * other.data_;
}

void EigenTensor::multiply_GEMM(const EigenTensor& other, EigenTensor& result) const {
    // Interpret:
    // this->data_  : M x K x D
    // other.data_  : K x N x D
    // result.data_ : M x N x D

    const int M = rows_;
    const int K = cols_;
    const int D = depth_;

    const int K2 = other.rows_;
    const int N  = other.cols_;
    const int D2 = other.depth_;

    if (K != K2 || D != D2) {
        throw std::runtime_error("EigenTensor::multiply: shape mismatch");
    }

    if (result.rows_ != M || result.cols_ != N || result.depth_ != D) {
        throw std::runtime_error("EigenTensor::multiply: result has wrong shape");
    }

    // Contract along K: A(M,K) * B(K,N) -> C(M,N)
    // For Eigen::Tensor, contract dims are given as (lhs_dim, rhs_dim).
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, 0)  // A.dim(1) with B.dim(0)  => K
    };

    for (int d = 0; d < D; ++d) {
        // 2D slices at fixed depth d:
        auto A = data_.chip(d, 2);        // shape: (M x K)
        auto B = other.data_.chip(d, 2);  // shape: (K x N)

        // C_d = A_d * B_d
        result.data_.chip(d, 2) = A.contract(B, contract_dims);
    }
}