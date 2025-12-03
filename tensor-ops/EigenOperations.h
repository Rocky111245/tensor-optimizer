#ifndef EIGENOPERATIONS_H
#define EIGENOPERATIONS_H

#include <unsupported/Eigen/CXX11/Tensor>

class EigenTensor {
public:
    EigenTensor(int rows, int cols, int depth);

    void randomize(float min, float max);
    void multiply(const EigenTensor& other, EigenTensor& result) const;
    void multiply_GEMM(const EigenTensor& other, EigenTensor& result) const;

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int depth() const { return depth_; }

private:
    int rows_, cols_, depth_;
    Eigen::Tensor<float, 3> data_;
};

#endif // EIGENOPERATIONS_H
