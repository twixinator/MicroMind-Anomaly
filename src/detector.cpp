#include "micromind/detector.h"
#include <array>
#include <cstdint>

namespace micromind {

void Detector::register_callback(StopCallback cb) {
    callback_ = cb;
}

void Detector::push_sensor_value(float value) {
    buffer_.push(value);

    // ---------------------------------------------------------------------------
    // Build the INPUT_FEATURES-element input vector from the ring buffer.
    //
    // We always want the most recent INPUT_FEATURES samples in chronological
    // order (oldest first), which is what the inference engine expects.
    //
    // When the buffer holds fewer than INPUT_FEATURES samples, the leading
    // slots are zero-padded so the network sees a consistent input shape.
    // ---------------------------------------------------------------------------
    std::array<float, INPUT_FEATURES> input{};

    uint32_t available = buffer_.size();

    // 'start' is the logical index of the oldest sample we want to copy.
    // If available >= INPUT_FEATURES we skip the oldest (available - INPUT_FEATURES)
    // samples and begin at that offset; otherwise we start at 0 and the
    // remaining slots stay at zero (from the value-initialisation above).
    uint32_t start = (available >= INPUT_FEATURES)
                         ? (available - INPUT_FEATURES)
                         : 0u;

    for (uint32_t i = 0u; i < INPUT_FEATURES; ++i) {
        uint32_t idx = start + i;   // both operands are uint32_t — no signed/unsigned warning
        if (idx < available) {
            input[i] = buffer_[idx];
        }
        // else: input[i] stays 0.0f (zero-padding)
    }

    // ---------------------------------------------------------------------------
    // Dummy 1-output dense layer.
    //
    // weights_ is a function-local static const so it lives in .rodata with no
    // dynamic allocation and no C++14 ODR out-of-class definition required.
    // All weights are 0.25f: with INPUT_FEATURES=4 inputs at full scale this
    // yields a max raw output of 1.0f, which is above ANOMALY_THRESHOLD=0.5f,
    // giving a testable range for integration tests.
    //
    // Replace this array with real trained weights once calibrated on hardware.
    // ---------------------------------------------------------------------------
    static const float weights[OUTPUT_FEATURES][INPUT_FEATURES] = {
        { 0.25f, 0.25f, 0.25f, 0.25f }
    };

    std::array<float, OUTPUT_FEATURES> output{};
    mat_vec_mul(weights, input, output);
    relu(output);

    // Fire the callback synchronously if an anomaly is detected.
    // The null-check avoids a branch-to-null fault on targets with no MMU.
    if (callback_ != nullptr && output[0] > ANOMALY_THRESHOLD) {
        callback_();
    }
}

} // namespace micromind
