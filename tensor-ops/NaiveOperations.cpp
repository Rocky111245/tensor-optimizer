//
// Created by Rocky170 on 11/25/2025.
//Simplest possible multiplication

#include "NaiveOperations.h"
#include <random>

NaiveTensor::NaiveTensor(int rows, int cols, int depth)
    : rows_(rows), cols_(cols), depth_(depth) {
    data_.resize(rows * cols * depth);
}

void NaiveTensor::randomize(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (auto& val : data_) {
        val = dist(gen);
    }
}

void NaiveTensor::multiply(const NaiveTensor& other, NaiveTensor& result) const {
    // Element-wise multiplication (Hadamard product)
    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            for (int d = 0; d < depth_; ++d) {
                result.at(r, c, d) = this->at(r, c, d) * other.at(r, c, d);
            }
        }
    }
}