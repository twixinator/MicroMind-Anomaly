#include <gtest/gtest.h>
#include "micromind/inference_engine.h"

// ---------------------------------------------------------------------------
// mat_vec_mul tests
// ---------------------------------------------------------------------------

// Identity matrix * v should reproduce v exactly (no floating-point error).
TEST(MatVecMul, IdentityMatrix)
{
    const float identity[2][2] = {
        {1.0f, 0.0f},
        {0.0f, 1.0f}
    };
    const std::array<float, 2> vec  = {3.0f, 4.0f};
    std::array<float, 2>       out  = {};

    micromind::mat_vec_mul(identity, vec, out);

    EXPECT_FLOAT_EQ(out[0], 3.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
}

// [[1,2],[3,4]] * [1,1] == [3, 7]
TEST(MatVecMul, KnownValues)
{
    const float matrix[2][2] = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };
    const std::array<float, 2> vec = {1.0f, 1.0f};
    std::array<float, 2>       out = {};

    micromind::mat_vec_mul(matrix, vec, out);

    EXPECT_FLOAT_EQ(out[0], 3.0f);
    EXPECT_FLOAT_EQ(out[1], 7.0f);
}

// Verify that out is zeroed before accumulation, even when the caller
// pre-fills it with garbage values.
TEST(MatVecMul, OutputIsZeroedBeforeAccumulation)
{
    const float matrix[2][2] = {
        {1.0f, 0.0f},
        {0.0f, 1.0f}
    };
    const std::array<float, 2> vec      = {5.0f, 6.0f};
    // Pre-fill with non-zero stale data to confirm zeroing.
    std::array<float, 2>       out      = {99.0f, 99.0f};

    micromind::mat_vec_mul(matrix, vec, out);

    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], 6.0f);
}

// Non-square matrix: 3x2 matrix * 2-vector -> 3-vector.
TEST(MatVecMul, NonSquareMatrix)
{
    // [[1,0],[0,1],[1,1]] * [2,3] == [2, 3, 5]
    const float matrix[3][2] = {
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f}
    };
    const std::array<float, 2> vec = {2.0f, 3.0f};
    std::array<float, 3>       out = {};

    micromind::mat_vec_mul(matrix, vec, out);

    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 3.0f);
    EXPECT_FLOAT_EQ(out[2], 5.0f);
}

// ---------------------------------------------------------------------------
// relu tests
// ---------------------------------------------------------------------------

// All-negative inputs must be clamped to 0.
TEST(Relu, AllNegative)
{
    std::array<float, 3> arr = {-1.0f, -2.0f, -3.0f};
    micromind::relu(arr);

    EXPECT_FLOAT_EQ(arr[0], 0.0f);
    EXPECT_FLOAT_EQ(arr[1], 0.0f);
    EXPECT_FLOAT_EQ(arr[2], 0.0f);
}

// Mixed: negative stays 0, zero stays 0, positive stays positive.
TEST(Relu, Mixed)
{
    std::array<float, 3> arr = {-1.0f, 0.0f, 2.0f};
    micromind::relu(arr);

    EXPECT_FLOAT_EQ(arr[0], 0.0f);
    EXPECT_FLOAT_EQ(arr[1], 0.0f);
    EXPECT_FLOAT_EQ(arr[2], 2.0f);
}

// All-positive inputs must pass through unchanged.
TEST(Relu, AllPositive)
{
    std::array<float, 3> arr = {1.0f, 2.0f, 3.0f};
    micromind::relu(arr);

    EXPECT_FLOAT_EQ(arr[0], 1.0f);
    EXPECT_FLOAT_EQ(arr[1], 2.0f);
    EXPECT_FLOAT_EQ(arr[2], 3.0f);
}

// ---------------------------------------------------------------------------
// dense_forward tests
// ---------------------------------------------------------------------------

// ReLU activation clamps negative pre-activation values to zero.
// weights = [[1, -2]], bias = [0], input = [1, 1]
// pre-activation = 1*1 + (-2)*1 + 0 = -1.0 -> ReLU -> 0.0
TEST(DenseForward, ReluActivation_ClampsNegatives)
{
    const float weights[1][2] = { {1.0f, -2.0f} };
    const std::array<float, 1> bias = {0.0f};
    const std::array<float, 2> input = {1.0f, 1.0f};
    std::array<float, 1> output = {99.0f};

    micromind::dense_forward(weights, bias, input, output,
                             micromind::Activation::kRelu);

    EXPECT_FLOAT_EQ(output[0], 0.0f);
}

// No activation passes raw values through, including negatives.
// weights = [[1, -2]], bias = [0], input = [1, 1]
// pre-activation = -1.0 -> kNone -> -1.0
TEST(DenseForward, NoActivation_PassesThrough)
{
    const float weights[1][2] = { {1.0f, -2.0f} };
    const std::array<float, 1> bias = {0.0f};
    const std::array<float, 2> input = {1.0f, 1.0f};
    std::array<float, 1> output = {99.0f};

    micromind::dense_forward(weights, bias, input, output,
                             micromind::Activation::kNone);

    EXPECT_FLOAT_EQ(output[0], -1.0f);
}

// Non-zero bias shifts the output by the bias value.
// weights = [[1, 1]], bias = [0.5], input = [2, 3]
// mat_mul = 1*2 + 1*3 = 5.0, bias = 0.5, output = 5.0 + 0.5 = 5.5
TEST(DenseForward, BiasCorrectness_ShiftsOutput)
{
    const float weights[1][2] = { {1.0f, 1.0f} };
    const std::array<float, 1> bias = {0.5f};
    const std::array<float, 2> input = {2.0f, 3.0f};
    std::array<float, 1> output = {99.0f};

    micromind::dense_forward(weights, bias, input, output,
                             micromind::Activation::kNone);

    EXPECT_FLOAT_EQ(output[0], 5.5f);
}

// Full two-layer pass matching the spec's hand-calculated values.
// Layer 1: 8x4 weights all 0.25, bias all 0.0, input [1,1,1,1]
//   each hidden unit = 0.25*4 = 1.0 -> ReLU -> 1.0
// Layer 2: 1x8 weights all 0.125, bias all 0.0
//   output = 0.125*8 = 1.0
TEST(DenseForward, KnownTwoLayerPass_MatchesHandCalc)
{
    const float w1[8][4] = {
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f}
    };
    const std::array<float, 8> b1 = {};
    const float w2[1][8] = {
        {0.125f, 0.125f, 0.125f, 0.125f,
         0.125f, 0.125f, 0.125f, 0.125f}
    };
    const std::array<float, 1> b2 = {};

    const std::array<float, 4> input = {1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, 8> hidden = {};
    std::array<float, 1> output = {};

    micromind::dense_forward(w1, b1, input, hidden,
                             micromind::Activation::kRelu);
    micromind::dense_forward(w2, b2, hidden, output,
                             micromind::Activation::kNone);

    for (std::size_t i = 0; i < hidden.size(); ++i) {
        EXPECT_FLOAT_EQ(hidden[i], 1.0f);
    }
    EXPECT_FLOAT_EQ(output[0], 1.0f);
}

// Two-layer pass with mixed-sign layer 1 weights: negative hidden units
// are clamped by ReLU and must not contribute to the output.
//
// Layer 1 (2x2): weights = [[1, 1], [-1, -1]], bias = [0, 0]
//   input = [1, 1]
//   hidden[0] = 1+1 = 2.0 -> ReLU -> 2.0
//   hidden[1] = -1-1 = -2.0 -> ReLU -> 0.0
// Layer 2 (1x2): weights = [[1, 1]], bias = [0]
//   output = 1*2.0 + 1*0.0 = 2.0 (NOT 0.0 if ReLU were missing)
TEST(DenseForward, NegativeHiddenUnits_ClampedByRelu)
{
    const float w1[2][2] = {
        { 1.0f,  1.0f},
        {-1.0f, -1.0f}
    };
    const std::array<float, 2> b1 = {};
    const float w2[1][2] = {
        {1.0f, 1.0f}
    };
    const std::array<float, 1> b2 = {};

    const std::array<float, 2> input = {1.0f, 1.0f};
    std::array<float, 2> hidden = {};
    std::array<float, 1> output = {};

    micromind::dense_forward(w1, b1, input, hidden,
                             micromind::Activation::kRelu);
    micromind::dense_forward(w2, b2, hidden, output,
                             micromind::Activation::kNone);

    EXPECT_FLOAT_EQ(hidden[0], 2.0f);
    EXPECT_FLOAT_EQ(hidden[1], 0.0f);
    EXPECT_FLOAT_EQ(output[0], 2.0f);
}
