#pragma once
#include "config.h"
#include "ring_buffer.h"
#include "inference_engine.h"
#include <cstdint>

namespace micromind {

// StopCallback is a raw function pointer (no std::function — avoids heap
// allocation and exception-based type erasure, both prohibited on this target).
using StopCallback = void(*)();

// ---------------------------------------------------------------------------
// Detector
//
// Accumulates sensor values in a fixed-capacity ring buffer and runs a
// two-layer dense inference pass every time a new sample arrives:
//   Layer 1: INPUT_FEATURES -> HIDDEN_UNITS (ReLU)
//   Layer 2: HIDDEN_UNITS   -> OUTPUT_FEATURES (no activation)
// When the output exceeds ANOMALY_THRESHOLD the registered StopCallback
// is fired.
//
// Memory model: all storage is static.  No heap, no exceptions.
// Thread safety: none — single-threaded embedded use only.
// ---------------------------------------------------------------------------
class Detector {
public:
    // Registers a callback that is invoked when an anomaly is detected.
    // Passing nullptr clears a previously registered callback.
    void register_callback(StopCallback cb);

    // Pushes a new sensor sample, then runs a forward pass over the last
    // INPUT_FEATURES samples (zero-padded if fewer are available).
    // If the output exceeds ANOMALY_THRESHOLD and a callback is registered,
    // the callback is called synchronously before this function returns.
    void push_sensor_value(float value);

private:
    RingBuffer<float, RING_BUFFER_CAPACITY> buffer_;
    StopCallback callback_ = nullptr;

    // Weights and biases are function-local static const inside
    // push_sensor_value() to avoid C++14 ODR issues and ensure .rodata
    // placement. See detector.cpp for values.
};

} // namespace micromind
