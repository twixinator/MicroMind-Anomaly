#pragma once
#include <cstddef>
#include <cstdint>

namespace micromind {
    static constexpr uint32_t    RING_BUFFER_CAPACITY = 32;
    static constexpr float       ANOMALY_THRESHOLD    = 0.5f;
    static constexpr std::size_t INPUT_FEATURES       = 4;
    static constexpr std::size_t OUTPUT_FEATURES      = 1;
    static constexpr std::size_t HIDDEN_UNITS         = 8;
    static constexpr float       LAYER1_BIAS_INIT     = 0.0f;
    static constexpr float       LAYER2_BIAS_INIT     = 0.0f;

    static_assert(HIDDEN_UNITS <= 64,
                  "HIDDEN_UNITS exceeds safe stack budget for embedded targets");
    static_assert(OUTPUT_FEATURES == 1,
                  "Detector callback logic assumes single-output inference");
} // namespace micromind
