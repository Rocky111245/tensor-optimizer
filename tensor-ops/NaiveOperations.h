//
// Created by Rocky170 on 11/25/2025.
//

#ifndef NAIVEOPERATIONS_H
#define NAIVEOPERATIONS_H

#include <vector>

class NaiveTensor {
public:
    NaiveTensor(int rows, int cols, int depth);

    void randomize(float min, float max);
    void multiply(const NaiveTensor& other, NaiveTensor& result) const;

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int depth() const { return depth_; }

private:
    int rows_, cols_, depth_;
    std::vector<float> data_;

    float& at(int r, int c, int d) { return data_[r * cols_ * depth_ + c * depth_ + d]; }
    const float& at(int r, int c, int d) const { return data_[r * cols_ * depth_ + c * depth_ + d]; }
};

#endif //NAIVEOPERATIONS_H
