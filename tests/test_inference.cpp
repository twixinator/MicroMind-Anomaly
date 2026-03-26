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
