#include "micromind/detector.h"
#include <array>
#include <cmath>
#include <cstdint>

namespace micromind {

void Detector::register_callback(StopCallback cb) {
    callback_ = cb;
}

void Detector::push_sensor_value(float value) {
    // Reentrance guard: calling push_sensor_value from within the
    // StopCallback would cause unbounded recursion on a small stack.
    if (in_push_) { return; }
    in_push_ = true;

    // Reject non-finite sensor values at the boundary. A NaN or Inf
    // from a malfunctioning ADC would silently disable detection
    // (NaN > threshold is always false). Treat as immediate anomaly.
    if (!std::isfinite(value)) {
        if (callback_ != nullptr) {
            callback_();
        }
        in_push_ = false;
        return;
    }

    buffer_.push(value);

    // ---------------------------------------------------------------------------
    // Build the INPUT_FEATURES-element input vector from the ring buffer.
    //
    // We always want the most recent INPUT_FEATURES samples in chronological
    // order (oldest first), which is what the inference engine expects.
    //
    // When the buffer holds fewer than INPUT_FEATURES samples, the trailing
    // slots are zero-padded so the network sees a consistent input shape.
    // Samples are left-aligned: [v, 0, 0, 0] for a single-sample cold start.
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

    // Numeric fault check: if inference produced NaN or Inf (e.g., from
    // overflow with trained weights), treat as anomaly rather than silently
    // missing the detection.
    if (!std::isfinite(output[0])) {
        if (callback_ != nullptr) {
            callback_();
        }
        in_push_ = false;
        return;
    }

    // Fire the callback synchronously if an anomaly is detected.
    // The null-check avoids a branch-to-null fault on targets with no MMU.
    if (callback_ != nullptr && output[0] > ANOMALY_THRESHOLD) {
        callback_();
    }

    in_push_ = false;
}

} // namespace micromind
