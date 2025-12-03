#ifndef NATIVEOPERATIONS_H
#define NATIVEOPERATIONS_H


#include "native-tensor-library/NativeTensorLibrary.h"

class NativeTensorWrapper {
public:
    NativeTensorWrapper(int rows, int cols, int depth);

    void randomize(float min, float max);
    void multiplyElementwise(const NativeTensorWrapper& other, NativeTensorWrapper& result) const;
    void multiplyElementwiseUnrolled(const NativeTensorWrapper& other, NativeTensorWrapper& result) const;

    //  SIMD version
    void multiplyElementwiseSimd(const NativeTensorWrapper& other, NativeTensorWrapper& result) const;

private:
    Tensor tensor_;
};

#endif // NATIVEOPERATIONS_H
