#include <gtest/gtest.h>
#include "micromind/detector.h"

// ---------------------------------------------------------------------------
// Shared callback infrastructure.
// Raw function pointers are required (no std::function per project rules).
// A file-scope flag is the only allocation-free way to observe callback
// invocation across a C function pointer boundary.
// ---------------------------------------------------------------------------
static bool g_callback_fired = false;

static void recording_callback() {
    g_callback_fired = true;
}

// Reset the flag before each test that uses the callback.
static void reset_callback() {
    g_callback_fired = false;
}

// ---------------------------------------------------------------------------
// Detector tests
// ---------------------------------------------------------------------------

// When the ring buffer contains INPUT_FEATURES samples all at 1.0f,
// the forward pass output (1.0f) exceeds ANOMALY_THRESHOLD (0.5f)
// and the registered callback must be invoked.
TEST(Detector, CallbackFiresAboveThreshold) {
    reset_callback();
    micromind::Detector det;
    det.register_callback(recording_callback);

    // Push INPUT_FEATURES (4) samples large enough to exceed threshold.
    // output = 0.25 * (1+1+1+1) = 1.0f > 0.5f
    for (uint32_t i = 0; i < micromind::INPUT_FEATURES; ++i) {
        det.push_sensor_value(1.0f);
    }

    EXPECT_TRUE(g_callback_fired);
}

// When sensor values produce an output below ANOMALY_THRESHOLD,
// the callback must NOT be invoked.
TEST(Detector, CallbackDoesNotFireBelowThreshold) {
    reset_callback();
    micromind::Detector det;
    det.register_callback(recording_callback);

    // output = 0.25 * (0+0+0+0) = 0.0f < 0.5f
    for (uint32_t i = 0; i < micromind::INPUT_FEATURES; ++i) {
        det.push_sensor_value(0.0f);
    }

    EXPECT_FALSE(g_callback_fired);
}

// Registering a null callback and then pushing values that would normally
// trigger a callback must not crash (null-pointer dereference check).
TEST(Detector, NullCallbackIsSafe) {
    micromind::Detector det;
    det.register_callback(nullptr);

    // Would fire if a callback were registered.
    for (uint32_t i = 0; i < micromind::INPUT_FEATURES; ++i) {
        det.push_sensor_value(1.0f);
    }
    // Test passes if we reach here without a crash.
    SUCCEED();
}

// On cold start (fewer than INPUT_FEATURES samples pushed), the missing
// slots are zero-padded. With only one sample at 1.0f the output is
// 0.25 * 1.0 = 0.25f which is below threshold — no callback.
TEST(Detector, ZeroPaddingOnColdStart) {
    reset_callback();
    micromind::Detector det;
    det.register_callback(recording_callback);

    // Push only one sample — three slots zero-padded.
    // output = 0.25 * 1.0f = 0.25f < 0.5f
    det.push_sensor_value(1.0f);

    EXPECT_FALSE(g_callback_fired);
}
