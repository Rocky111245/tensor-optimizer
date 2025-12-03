//
// Created by Rocky170 on 11/26/2025.
//

#include "ETensor.h"

#include <immintrin.h>
#include <cmath>
#include <vector>


/* ==============================  Constructors  ============================== */

ETensor::ETensor() noexcept = default;

ETensor::ETensor(int rows, int columns, int depth)
        : rows_(rows), columns_(columns), depth_(depth), data_(std::make_unique<float[]>(rows * columns * depth)) {
    if (rows <= 0 || columns <= 0 || depth <= 0)
        throw std::invalid_argument("Tensor dimensions must be positive in main Tensor constructor.");
    std::fill(data_.get(), data_.get() + rows * columns * depth, 0.0f);
}

ETensor::ETensor(const ETensor& other)  // Copy constructor
        : rows_(other.rows_), columns_(other.columns_), depth_(other.depth_),
          data_(std::make_unique<float[]>(rows_ * columns_ * depth_)) {
    std::copy(other.data_.get(),
              other.data_.get() + rows_ * columns_ * depth_,
              data_.get());
}

ETensor::ETensor(ETensor&& other) noexcept  // Move constructor
        : rows_(other.rows_), columns_(other.columns_), depth_(other.depth_),
          data_(std::move(other.data_)) {
    other.rows_ = other.columns_ = other.depth_ = 0;
}

/* ==============================  Assignment Operators  ============================== */

ETensor& ETensor::operator=(const ETensor& other) {  // Copy assignment
    if (this != &other) {
        if (rows_ != other.rows_ || columns_ != other.columns_ || depth_ != other.depth_) {
            rows_ = other.rows_;
            columns_ = other.columns_;
            depth_ = other.depth_;
            data_ = std::make_unique<float[]>(rows_ * columns_ * depth_);
        }
        std::copy(other.data_.get(),
                  other.data_.get() + rows_ * columns_ * depth_,
                  data_.get());
    }
    return *this;
}

ETensor& ETensor::operator=(ETensor&& other) noexcept {  // Move assignment
    if (this != &other) {
        rows_ = other.rows_;
        columns_ = other.columns_;
        depth_ = other.depth_;
        data_ = std::move(other.data_);
        other.rows_ = other.columns_ = other.depth_ = 0;
    }
    return *this;
}

/* ==============================  Operator Overloads  ============================== */

bool ETensor::operator==(const ETensor& other) const {
    constexpr float epsilon = 1e-5f;

    if (rows_ != other.rows_ || columns_ != other.columns_ || depth_ != other.depth_)
        return false;

    int total_elements = rows_ * columns_ * depth_;
    for (int i = 0; i < total_elements; ++i) {
        if (std::fabs(data_[i] - other.data_[i]) > epsilon)
            return false;
    }
    return true;
}


ETensor& ETensor::operator+=(const ETensor& other) {
    if(this->rows() != other.rows() || this->columns() != other.columns() || this->depth() != other.depth()) {
        throw std::invalid_argument("Tensors must have exactly same shape");
    }

    int depth = this->depth();
    int row = this->rows();
    int column = this->columns();
    for (int d = 0; d < depth; d++) {
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                (*this)(r, c, d) += other(r, c, d);
            }
        }
    }

    return *this;
}


/* ==============================  Element Access Operators  ============================== */

const float& ETensor::operator()(int row, int column, int depth) const noexcept {  // Read access
    return data_[Index(row, column, depth)];
}

float& ETensor::operator()(int row, int column, int depth) noexcept {  // Write access
    return data_[Index(row, column, depth)];
}


const float* ETensor::Get_Data() const noexcept {
    return data_.get();
}

// For non-const tensors - returns read-write access
float* ETensor::Get_Data() noexcept {
    return data_.get();
}

void Tensor_Transpose(ETensor& result, const ETensor& input) {
    // Input: [rows, columns, depth] -> Output: [columns, rows, depth]

    if (result.rows() != input.columns() ||
        result.columns() != input.rows() ||
        result.depth() != input.depth()) {
        throw std::invalid_argument("Result tensor dimensions must match transposed input dimensions");
        }

    // Transpose each batch channel independently
    for (int batch_idx = 0; batch_idx < input.depth(); ++batch_idx) {
        for (int row = 0; row < input.rows(); ++row) {
            for (int col = 0; col < input.columns(); ++col) {
                // Swap row and column indices for transpose
                result(col, row, batch_idx) = input(row, col, batch_idx);
            }
        }
    }
}


// The rows represents the neurones, the columns will represent the number of features.
// Column is fastest moving (features), depth is slowest moving

//input is (4x3x2),(3x4x2), so result will be (4x4x2).
// So next layer will have 4 neurones.
// ETensor.cpp

void Tensor_Multiply_Tensor(ETensor& result,
                            const ETensor& first,
                            const ETensor& second) {
    // Shapes:
    // first  : M x K x D
    // second : K x N x D
    // result : M x N x D

    const int M = first.rows();
    const int K = first.columns();
    const int D = first.depth();

    const int K2 = second.rows();
    const int N  = second.columns();
    const int D2 = second.depth();

    if (K != K2 || D != D2) {
        throw std::runtime_error("Tensor_Multiply_Tensor: shape mismatch");
    }


    if (result.rows() != M || result.columns() != N || result.depth() != D) {
        throw std::runtime_error("Tensor_Multiply_Tensor: result has wrong shape");
    }

    //Transpose
    //    temporary: N x K x D
    ETensor temporary(second.columns(), second.rows(), second.depth());
    Tensor_Transpose(temporary, second);  // temporary(j,k,d) = second(k,j,d)

    const float* first_data  = first.Get_Data();       // A
    const float* temp_data   = temporary.Get_Data();   // B^T
    float*       result_data = result.Get_Data();      // C

    const int strideA_d  = M * K;  // elements per depth slice of A
    const int strideBT_d = N * K;  // elements per depth slice of B^T
    const int strideC_d  = M * N;  // elements per depth slice of C


    std::fill(result_data, result_data + D * strideC_d, 0.0f);

    // 2) For each depth slice, do matmul by row·row dot products:
    for (int d = 0; d < D; ++d) {
        const float* Ad  = first_data  + d * strideA_d;
        const float* BTd = temp_data   + d * strideBT_d;
        float*       Cd  = result_data + d * strideC_d;

        for (int i = 0; i < M; ++i) {
            const float* Arow = Ad + i * K;   // row i of A_d (length K)

            for (int j = 0; j < N; ++j) {
                const float* Brow = BTd + j * K;  // row j of B^T_d (length K)
                float sum = 0.0f;

                // Inner loop: dot product over K, fully contiguous in Arow and Brow
                int k = 0;
                const int limit = K & ~3; // unroll by 4

                for (; k < limit; k += 4) {
                    sum += Arow[k + 0] * Brow[k + 0];
                    sum += Arow[k + 1] * Brow[k + 1];
                    sum += Arow[k + 2] * Brow[k + 2];
                    sum += Arow[k + 3] * Brow[k + 3];
                }
                for (; k < K; ++k) {
                    sum += Arow[k] * Brow[k];
                }

                // C(i, j, d)
                Cd[i * N + j] = sum;
            }
        }
    }
}

//--------------------------------Optimized Multiply Below------------------------------------------//
// Horizontal sum helper
inline float hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    vlow = _mm_hadd_ps(vlow, vlow);
    vlow = _mm_hadd_ps(vlow, vlow);
    return _mm_cvtss_f32(vlow);
}

void Tensor_Multiply_Tensor_SIMD(ETensor& result,
                                          const ETensor& first,
                                          const ETensor& second) {
    // Shapes:
    // first  (A): M x K x D
    // second (B): K x N x D
    // result (C): M x N x D

    const int M = first.rows();
    const int K = first.columns();
    const int D = first.depth();
    const int N = second.columns();

    const float* A_data = first.Get_Data();
    const float* B_data = second.Get_Data();
    float* C_data = result.Get_Data();

    const int strideA = M * K;
    const int strideB = K * N;
    const int strideC = M * N;

#if defined(__AVX2__)

    // ============================================================
    // STEP 1: Allocate buffer for transposed B (N x K per depth)
    // ============================================================
    std::vector<float> B_transposed(N * K);

    for (int d = 0; d < D; ++d) {
        const float* Ad = A_data + d * strideA;
        const float* Bd = B_data + d * strideB;
        float* Cd = C_data + d * strideC;

        // ============================================================
        // STEP 2: Transpose B (K x N) -> B_transposed (N x K)
        // This has strided READS but sequential WRITES
        // ============================================================
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                // Read: strided (jumping by N each k)
                // Write: sequential (n * K + k is contiguous for fixed n)
                B_transposed[n * K + k] = Bd[k * N + n];
            }
        }

        // ============================================================
        // STEP 3: Sequential access on B_transposed
        // ============================================================
        const float* B_trans_ptr = B_transposed.data();

        for (int i = 0; i < M; ++i) {
            const float* row_A = Ad + i * K;

            int j = 0;
            // Unrolled loop: Process 4 rows of B_transposed at once
            for (; j <= N - 4; j += 4) {
                const float* row_B0 = B_trans_ptr + (j + 0) * K;
                const float* row_B1 = B_trans_ptr + (j + 1) * K;
                const float* row_B2 = B_trans_ptr + (j + 2) * K;
                const float* row_B3 = B_trans_ptr + (j + 3) * K;

                __m256 sum0 = _mm256_setzero_ps();
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m256 sum3 = _mm256_setzero_ps();

                int k = 0;
                for (; k <= K - 8; k += 8) {
                    __m256 va = _mm256_loadu_ps(row_A + k);

                    __m256 vb0 = _mm256_loadu_ps(row_B0 + k);
                    __m256 vb1 = _mm256_loadu_ps(row_B1 + k);
                    __m256 vb2 = _mm256_loadu_ps(row_B2 + k);
                    __m256 vb3 = _mm256_loadu_ps(row_B3 + k);

                    sum0 = _mm256_fmadd_ps(va, vb0, sum0);
                    sum1 = _mm256_fmadd_ps(va, vb1, sum1);
                    sum2 = _mm256_fmadd_ps(va, vb2, sum2);
                    sum3 = _mm256_fmadd_ps(va, vb3, sum3);
                }

                float val0 = hsum256_ps_avx(sum0);
                float val1 = hsum256_ps_avx(sum1);
                float val2 = hsum256_ps_avx(sum2);
                float val3 = hsum256_ps_avx(sum3);

                for (; k < K; ++k) {
                    float a_val = row_A[k];
                    val0 += a_val * row_B0[k];
                    val1 += a_val * row_B1[k];
                    val2 += a_val * row_B2[k];
                    val3 += a_val * row_B3[k];
                }

                Cd[i * N + (j + 0)] = val0;
                Cd[i * N + (j + 1)] = val1;
                Cd[i * N + (j + 2)] = val2;
                Cd[i * N + (j + 3)] = val3;
            }

            // Cleanup remaining columns
            for (; j < N; ++j) {
                const float* row_B = B_trans_ptr + j * K;
                __m256 sum = _mm256_setzero_ps();
                int k = 0;
                for (; k <= K - 8; k += 8) {
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(row_A + k),
                                          _mm256_loadu_ps(row_B + k), sum);
                }
                float val = hsum256_ps_avx(sum);
                for (; k < K; ++k) val += row_A[k] * row_B[k];
                Cd[i * N + j] = val;
            }
        }
    }

#else
    // Scalar fallback (same as before)
    for (int d = 0; d < D; ++d) {
        const float* Ad = A_data + d * strideA;
        const float* Bd = B_data + d * strideB;
        float* Cd = C_data + d * strideC;

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += Ad[i * K + k] * Bd[k * N + j];
                }
                Cd[i * N + j] = sum;
            }
        }
    }
#endif
}
