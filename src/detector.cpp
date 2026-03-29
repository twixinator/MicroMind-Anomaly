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

    const uint32_t available = buffer_.size();

    // 'start' is the logical index of the oldest sample we want to copy.
    // If available >= INPUT_FEATURES we skip the oldest samples;
    // otherwise we start at 0 and remaining slots stay zero-padded.
    const uint32_t start = (available >= static_cast<uint32_t>(INPUT_FEATURES))
                               ? (available - static_cast<uint32_t>(INPUT_FEATURES))
                               : 0u;

    for (uint32_t i = 0u; i < static_cast<uint32_t>(INPUT_FEATURES); ++i) {
        const uint32_t idx = start + i;
        if (idx < available) {
            input[i] = buffer_[idx];
        }
        // else: input[i] stays 0.0f (zero-padding)
    }

    // ---------------------------------------------------------------------------
    // Two-layer forward pass.
    //
    // All weights and biases are function-local static const arrays placed in
    // .rodata at link time. Placeholder values use uniform reciprocals for
    // hand-computable test expectations. Replace with trained weights via the
    // export pipeline (v0.5).
    //
    // Layer 1: INPUT_FEATURES -> HIDDEN_UNITS, ReLU activation
    // Layer 2: HIDDEN_UNITS   -> OUTPUT_FEATURES, no activation (raw score)
    // ---------------------------------------------------------------------------
    static const float weights_l1[HIDDEN_UNITS][INPUT_FEATURES] = {
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f},
        {0.25f, 0.25f, 0.25f, 0.25f}
    };
    static const std::array<float, HIDDEN_UNITS> bias_l1 = {
        LAYER1_BIAS_INIT, LAYER1_BIAS_INIT, LAYER1_BIAS_INIT, LAYER1_BIAS_INIT,
        LAYER1_BIAS_INIT, LAYER1_BIAS_INIT, LAYER1_BIAS_INIT, LAYER1_BIAS_INIT
    };

    static const float weights_l2[OUTPUT_FEATURES][HIDDEN_UNITS] = {
        {0.125f, 0.125f, 0.125f, 0.125f,
         0.125f, 0.125f, 0.125f, 0.125f}
    };
    static const std::array<float, OUTPUT_FEATURES> bias_l2 = {LAYER2_BIAS_INIT};

    std::array<float, HIDDEN_UNITS> hidden{};
    std::array<float, OUTPUT_FEATURES> output{};

    dense_forward(weights_l1, bias_l1, input, hidden, Activation::kRelu);
    dense_forward(weights_l2, bias_l2, hidden, output, Activation::kNone);

    // Fire the callback synchronously if an anomaly is detected.
    // The null-check avoids a branch-to-null fault on targets with no MMU.
    if (callback_ != nullptr && output[0] > ANOMALY_THRESHOLD) {
        callback_();
    }
}

} // namespace micromind
