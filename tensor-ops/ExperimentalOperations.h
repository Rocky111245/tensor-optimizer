//
// Created by Rocky170 on 11/27/2025.
//

#ifndef EXPERIMENTALOPERATIONS_H
#define EXPERIMENTALOPERATIONS_H
#include "experimental-tensor-library/ETensor.h"

class ETensorWrapper {
public:
    ETensorWrapper(int rows, int cols, int depth);

    void randomize(float min, float max);

    void multiplyMatmulSpecial(const ETensorWrapper& other,
                               ETensorWrapper& result) const;
    void multiplyMatmulSimd(const ETensorWrapper& other,
                        ETensorWrapper& result) const;


    const ETensor& tensor() const { return tensor_; }
    ETensor& tensor()             { return tensor_; }

private:
    ETensor tensor_;
};
#endif //EXPERIMENTALOPERATIONS_H
