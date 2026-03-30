#pragma once
#include <cstddef>
#include <algorithm>  // std::max
#include <array>

namespace micromind {

// ---------------------------------------------------------------------------
// mat_vec_mul
//
// Computes a dense layer forward pass:
//   out[r] = sum_c( matrix[r][c] * vec[c] )  for r in [0, Rows)
//
// out[r] is explicitly zeroed before accumulation so callers are not required
// to pre-initialise the output array.
//
// Template parameters use std::size_t to match std::array's size_type,
// enabling template argument deduction on 64-bit platforms.
// ---------------------------------------------------------------------------
template<std::size_t Rows, std::size_t Cols>
void mat_vec_mul(const float (&matrix)[Rows][Cols],
                 const std::array<float, Cols>& vec,
                 std::array<float, Rows>& out)
{
    for (std::size_t r = 0; r < Rows; ++r) {
        out[r] = 0.0f;
        for (std::size_t c = 0; c < Cols; ++c) {
            out[r] += matrix[r][c] * vec[c];
        }
    }
}

// ---------------------------------------------------------------------------
// relu
//
// Applies the Rectified Linear Unit activation in-place:
//   arr[i] = max(0.0f, arr[i])
//
// Uses std::max to let the compiler emit a branchless MAXSS/VMAXPS on x86/ARM
// rather than a conditional branch, which is preferable on deeply pipelined
// embedded cores.
// ---------------------------------------------------------------------------
template<std::size_t N>
void relu(std::array<float, N>& arr)
{
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = std::max(0.0f, arr[i]);
    }
}

// ---------------------------------------------------------------------------
// Activation
//
// Selects the activation function applied after the dense layer.
// kRelu  — clamp negatives to zero (hidden layers)
// kNone  — identity / pass-through (output layer)
// ---------------------------------------------------------------------------
enum class Activation { kRelu, kNone };

// ---------------------------------------------------------------------------
// dense_forward
//
// Composes a full dense layer pass: matrix-vector multiply, bias addition,
// and optional activation.
//
// Zeroing contract: output is zeroed by mat_vec_mul before accumulation.
// dense_forward does not zero independently — callers need not initialize
// the output array before calling.
//
// The activation branch is a runtime switch; each template instantiation
// at a concrete call site allows the compiler to eliminate the dead branch
// after inlining.
// ---------------------------------------------------------------------------
template<std::size_t Rows, std::size_t Cols>
void dense_forward(const float (&weights)[Rows][Cols],
                   const std::array<float, Rows>& bias,
                   const std::array<float, Cols>& input,
                   std::array<float, Rows>& output,
                   Activation act)
{
    mat_vec_mul(weights, input, output);

    for (std::size_t i = 0; i < Rows; ++i) {
        output[i] += bias[i];
    }

    switch (act) {
        case Activation::kRelu:
            relu(output);
            break;
        case Activation::kNone:
            break;
    }
}

} // namespace micromind
