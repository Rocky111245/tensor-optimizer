//
// Created by Rocky170 on 11/25/2025.
//

#include "NativeOperations.h"
#include <immintrin.h>
#include <random>


NativeTensorWrapper::NativeTensorWrapper(int rows, int cols, int depth)
    : tensor_(rows, cols, depth) {
}

void NativeTensorWrapper::randomize(float min, float max) {
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

//Unroll
void NativeTensorWrapper::multiplyElementwiseUnrolled(const NativeTensorWrapper& other, NativeTensorWrapper& result) const {
    int total = tensor_.rows() * tensor_.columns() * tensor_.depth();

    // Get direct pointer access for speed
    const float* a_data = this->tensor_.Get_Data();
    const float* b_data = other.tensor_.Get_Data();
    float* result_data = result.tensor_.Get_Data();

    // Process 8 elements at a time (unroll by 8)
    int i = 0;
    for (; i <= total - 8; i += 8) {
        result_data[i+0] = a_data[i+0] * b_data[i+0];
        result_data[i+1] = a_data[i+1] * b_data[i+1];
        result_data[i+2] = a_data[i+2] * b_data[i+2];
        result_data[i+3] = a_data[i+3] * b_data[i+3];
        result_data[i+4] = a_data[i+4] * b_data[i+4];
        result_data[i+5] = a_data[i+5] * b_data[i+5];
        result_data[i+6] = a_data[i+6] * b_data[i+6];
        result_data[i+7] = a_data[i+7] * b_data[i+7];
    }

    // Handle remaining elements (if any)
    for (; i < total; ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }
}



void NativeTensorWrapper::multiplyElementwiseSimd(const NativeTensorWrapper& other,
                                        NativeTensorWrapper& result) const {
    const int total =
        tensor_.rows() * tensor_.columns() * tensor_.depth();

    const float* a_data = this->tensor_.Get_Data();
    const float* b_data = other.tensor_.Get_Data();
    float* out_data     = result.tensor_.Get_Data();

    int i = 0;

#if defined(__AVX2__)
    constexpr int VEC_WIDTH = 8; // 8 floats
    for (; i + VEC_WIDTH <= total; i += VEC_WIDTH) {
        __m256 va = _mm256_loadu_ps(a_data + i);
        __m256 vb = _mm256_loadu_ps(b_data + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out_data + i, vc);
    }
#elif defined(__SSE2__)
    constexpr int VEC_WIDTH = 4; // 4 floats
    for (; i + VEC_WIDTH <= total; i += VEC_WIDTH) {
        __m128 va = _mm_loadu_ps(a_data + i);
        __m128 vb = _mm_loadu_ps(b_data + i);
        __m128 vc = _mm_mul_ps(va, vb);
        _mm_storeu_ps(out_data + i, vc);
    }
#endif

    //tail
    for (; i < total; ++i) {
        out_data[i] = a_data[i] * b_data[i];
    }
}








