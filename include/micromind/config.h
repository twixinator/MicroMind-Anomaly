#pragma once
#include <cstdint>

namespace micromind {
    static constexpr uint32_t RING_BUFFER_CAPACITY = 32;
    static constexpr float ANOMALY_THRESHOLD = 0.5f;
    static constexpr uint32_t INPUT_FEATURES  = 4;
    static constexpr uint32_t OUTPUT_FEATURES = 1;
} // namespace micromind
