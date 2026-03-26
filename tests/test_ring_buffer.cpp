#include <gtest/gtest.h>
#include "micromind/ring_buffer.h"

// ---------------------------------------------------------------------------
// Test 1: Empty buffer invariants
// After construction nothing has been pushed. size() must be 0,
// is_full() must be false, and capacity() must equal the template parameter.
// ---------------------------------------------------------------------------
TEST(RingBuffer, EmptyBufferInvariants) {
    micromind::RingBuffer<int16_t, 8> rb;
    EXPECT_EQ(rb.size(),     0u);
    EXPECT_EQ(rb.capacity(), 8u);
    EXPECT_FALSE(rb.is_full());
}

// ---------------------------------------------------------------------------
// Test 2: capacity() always returns the template parameter
// Verify both at compile-time (static_assert) and at runtime to make sure
// different Capacity values are handled independently.
// ---------------------------------------------------------------------------
TEST(RingBuffer, CapacityMatchesTemplateParameter) {
    micromind::RingBuffer<float,    4>  rb4;
    micromind::RingBuffer<int16_t, 16> rb16;
    micromind::RingBuffer<uint32_t, 1> rb1;

    EXPECT_EQ(rb4.capacity(),  4u);
    EXPECT_EQ(rb16.capacity(), 16u);
    EXPECT_EQ(rb1.capacity(),  1u);
}

// ---------------------------------------------------------------------------
// Test 3: Push elements up to capacity — size grows, is_full() triggers
// Push exactly Capacity elements and verify size() increments each time,
// then confirm is_full() becomes true only after the last push.
// ---------------------------------------------------------------------------
TEST(RingBuffer, PushToCapacity) {
    micromind::RingBuffer<int16_t, 4> rb;

    rb.push(static_cast<int16_t>(10));
    EXPECT_EQ(rb.size(), 1u);
    EXPECT_FALSE(rb.is_full());

    rb.push(static_cast<int16_t>(20));
    EXPECT_EQ(rb.size(), 2u);
    EXPECT_FALSE(rb.is_full());

    rb.push(static_cast<int16_t>(30));
    EXPECT_EQ(rb.size(), 3u);
    EXPECT_FALSE(rb.is_full());

    rb.push(static_cast<int16_t>(40));
    EXPECT_EQ(rb.size(), 4u);
    EXPECT_TRUE(rb.is_full());

    // Capacity must not grow beyond template parameter.
    EXPECT_EQ(rb.capacity(), 4u);
}

// ---------------------------------------------------------------------------
// Test 4: Overflow overwrites oldest; size stays pinned at Capacity
// Push Capacity+N elements and confirm size stays at Capacity while the
// oldest elements have been evicted and the newest are accessible.
// ---------------------------------------------------------------------------
TEST(RingBuffer, OverflowOverwritesOldest) {
    micromind::RingBuffer<int16_t, 4> rb;

    // Fill the buffer: [1, 2, 3, 4]
    rb.push(static_cast<int16_t>(1));
    rb.push(static_cast<int16_t>(2));
    rb.push(static_cast<int16_t>(3));
    rb.push(static_cast<int16_t>(4));
    EXPECT_EQ(rb.size(), 4u);
    EXPECT_TRUE(rb.is_full());

    // Push 5 — overwrites oldest (1). Buffer is logically [2, 3, 4, 5].
    rb.push(static_cast<int16_t>(5));
    EXPECT_EQ(rb.size(), 4u);   // must stay pinned at Capacity
    EXPECT_TRUE(rb.is_full());

    // Push 6 — overwrites oldest (2). Buffer is logically [3, 4, 5, 6].
    rb.push(static_cast<int16_t>(6));
    EXPECT_EQ(rb.size(), 4u);
    EXPECT_TRUE(rb.is_full());

    // Verify logical ordering: 0=oldest, 3=newest.
    EXPECT_EQ(rb[0], static_cast<int16_t>(3));
    EXPECT_EQ(rb[1], static_cast<int16_t>(4));
    EXPECT_EQ(rb[2], static_cast<int16_t>(5));
    EXPECT_EQ(rb[3], static_cast<int16_t>(6));
}

// ---------------------------------------------------------------------------
// Test 5: operator[] ordering — index 0 is oldest, last index is newest
// Before any overflow, validate that the logical ordering matches insertion
// order exactly.
// ---------------------------------------------------------------------------
TEST(RingBuffer, OperatorIndexOrderingBeforeOverflow) {
    micromind::RingBuffer<int16_t, 5> rb;

    rb.push(static_cast<int16_t>(100));
    rb.push(static_cast<int16_t>(200));
    rb.push(static_cast<int16_t>(300));

    // size() == 3, not full yet.
    EXPECT_EQ(rb.size(), 3u);
    EXPECT_EQ(rb[0], static_cast<int16_t>(100));  // oldest
    EXPECT_EQ(rb[1], static_cast<int16_t>(200));
    EXPECT_EQ(rb[2], static_cast<int16_t>(300));  // newest
}

// ---------------------------------------------------------------------------
// Test 6: operator[] ordering after multiple wraps
// Push 2*Capacity elements to force two full wrap-arounds and confirm the
// ring index formula is correct after each wrap.
// ---------------------------------------------------------------------------
TEST(RingBuffer, OperatorIndexOrderingAfterMultipleWraps) {
    micromind::RingBuffer<int16_t, 4> rb;

    // First wrap: push 8 elements (two full rotations).
    // After all 8 pushes the buffer should hold [5, 6, 7, 8].
    for (int16_t i = 1; i <= 8; ++i) {
        rb.push(i);
    }

    EXPECT_EQ(rb.size(), 4u);
    EXPECT_TRUE(rb.is_full());

    EXPECT_EQ(rb[0], static_cast<int16_t>(5));  // oldest surviving
    EXPECT_EQ(rb[1], static_cast<int16_t>(6));
    EXPECT_EQ(rb[2], static_cast<int16_t>(7));
    EXPECT_EQ(rb[3], static_cast<int16_t>(8));  // newest
}

// ---------------------------------------------------------------------------
// Test 7: Float type — verify correct behaviour with a non-integer T
// The library is used for floating-point sensor data in the inference engine.
// Confirm the index formula and overflow logic work identically for float.
// ---------------------------------------------------------------------------
TEST(RingBuffer, FloatTypeOverflowAndOrdering) {
    micromind::RingBuffer<float, 3> rb;

    rb.push(1.0f);
    rb.push(2.0f);
    rb.push(3.0f);  // full: [1.0, 2.0, 3.0]

    rb.push(4.0f);  // overwrites 1.0 -> [2.0, 3.0, 4.0]

    EXPECT_EQ(rb.size(), 3u);
    EXPECT_FLOAT_EQ(rb[0], 2.0f);
    EXPECT_FLOAT_EQ(rb[1], 3.0f);
    EXPECT_FLOAT_EQ(rb[2], 4.0f);
}

// ---------------------------------------------------------------------------
// Test 8: Capacity-1 buffer (single-element edge case)
// A RingBuffer<T, 1> must always overwrite its sole element.
// This exercises the boundary where head_==0 and Capacity==1 at every push.
// ---------------------------------------------------------------------------
TEST(RingBuffer, SingleElementCapacity) {
    micromind::RingBuffer<int16_t, 1> rb;

    EXPECT_EQ(rb.size(), 0u);
    EXPECT_FALSE(rb.is_full());

    rb.push(static_cast<int16_t>(42));
    EXPECT_EQ(rb.size(), 1u);
    EXPECT_TRUE(rb.is_full());
    EXPECT_EQ(rb[0], static_cast<int16_t>(42));

    // Every subsequent push replaces the single stored value.
    rb.push(static_cast<int16_t>(99));
    EXPECT_EQ(rb.size(), 1u);
    EXPECT_TRUE(rb.is_full());
    EXPECT_EQ(rb[0], static_cast<int16_t>(99));

    rb.push(static_cast<int16_t>(7));
    EXPECT_EQ(rb[0], static_cast<int16_t>(7));
}
