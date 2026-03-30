#pragma once
#include <cstddef>
#include <cstdint>

namespace micromind {
    constexpr uint32_t    RING_BUFFER_CAPACITY = 32;
    constexpr float       ANOMALY_THRESHOLD    = 0.5f;
    constexpr std::size_t INPUT_FEATURES       = 4;
    constexpr std::size_t OUTPUT_FEATURES      = 1;
    constexpr std::size_t HIDDEN_UNITS         = 8;
    constexpr float       LAYER1_BIAS_INIT     = 0.0f;
    constexpr float       LAYER2_BIAS_INIT     = 0.0f;

    static_assert(HIDDEN_UNITS >= 1,
                  "HIDDEN_UNITS must be at least 1");
    static_assert(HIDDEN_UNITS <= 64,
                  "HIDDEN_UNITS exceeds safe stack budget for embedded targets");
    static_assert(OUTPUT_FEATURES == 1,
                  "Detector callback logic assumes single-output inference");
    static_assert(RING_BUFFER_CAPACITY >= INPUT_FEATURES,
                  "Ring buffer must hold at least one full input window");
} // namespace micromind
