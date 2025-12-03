//
// Created by Rocky170 on 11/26/2025.
//

#ifndef ETENSOR_H
#define ETENSOR_H

/**
 *  Tensor – minimal 3‑D float container.
 *  Memory layout: depth (slow) → row → column (fast)
 */

#include <memory>       // std::unique_ptr

/* ==============================  Tensor Class  ============================== */
class ETensor {
public:
    /* ----------------------------- Constructors ----------------------------- */
    ETensor() noexcept;
    ETensor(int rows, int columns, int depth);
    ETensor(const ETensor& other);                    // Copy constructor
    ETensor(ETensor&& other) noexcept;               // Move constructor
    ~ETensor() = default;                           // Destructor

    /* ----------------------------- Assignment Operators ----------------------------- */
    ETensor& operator=(const ETensor& other);        // Copy assignment
    ETensor& operator=(ETensor&& other) noexcept;   // Move assignment

    /* ----------------------------- Comparison Operators ----------------------------- */
    bool operator==(const ETensor& other) const;


    ETensor& operator+=(const ETensor& other);
    /* ----------------------------- Element Access Operators ----------------------------- */
    const float& operator()(int row, int column, int depth) const noexcept;  // Read access
    float& operator()(int row, int column, int depth) noexcept;              // Write access

    /* ----------------------------- Dimension Getters ----------------------------- */
    int rows() const noexcept;
    int columns() const noexcept;
    int depth() const noexcept;
    const float* Get_Data() const noexcept;
    float* Get_Data() noexcept;

private:
    /* ----------------------------- Helper Functions ----------------------------- */

    //size of tensor is depth*row*column

    int Index(int row, int column, int depth_idx) const noexcept {
        return depth_idx * rows_ * columns_ + row * columns_ + column;
    }

    /* ----------------------------- Member Variables ----------------------------- */
    int rows_ = 0; //
    int columns_ = 0;
    int depth_ = 0;
    std::unique_ptr<float[]> data_;
};




/* ==============================  Standalone Utility Functions  ============================== */

/* ----------------------------- Tensor Multiplication ----------------------------- */
void Tensor_Multiply_Tensor(ETensor& result, const ETensor& first, const ETensor& second);
void Tensor_Transpose(ETensor& result, const ETensor& input);

void Tensor_Multiply_Tensor_SIMD(ETensor& result,const ETensor& first,const ETensor& second) ;


#endif //ETENSOR_H
