#pragma once
#include <cstdint>
#include <array>
#include <cassert>

namespace micromind {

// RingBuffer<T, Capacity> — fixed-capacity circular buffer with static storage.
//
// Design decisions:
//   - head_ always points to the next write slot (one-past-newest convention).
//   - count_ tracks how many valid elements are currently held.
//   - When full, push() silently overwrites the oldest element; size() stays
//     pinned at Capacity. This "lossy" semantic is correct for sliding-window
//     feature extraction on embedded sensor streams.
//   - Index 0 is the oldest element (chronological order), which matches the
//     expected input layout for the downstream inference engine.
//   - No dynamic allocation, no exceptions, no external dependencies. C++14.
//
// Index formula for operator[](i):
//   physical = (head_ + Capacity - count_ + i) % Capacity
//
//   When not full: head_ == count_, so physical = i.  Oldest is at slot 0.
//   When full:     count_ == Capacity, so oldest is at slot head_ (the slot
//                  about to be overwritten on the next push).

template<typename T, uint32_t Capacity>
class RingBuffer {
    static_assert(Capacity > 0u, "RingBuffer Capacity must be greater than zero");
public:
    // Push a value into the buffer.
    // If the buffer is already full the oldest element is silently overwritten.
    // Complexity: O(1).
    void push(T value) {
        buffer_[head_] = value;
        head_ = static_cast<uint32_t>((head_ + 1u) % Capacity);
        if (count_ < Capacity) {
            ++count_;
        }
        // When full, head_ advancing already "forgets" the oldest slot;
        // count_ stays at Capacity.
    }

    // Access element at logical index, where 0 is the oldest element.
    // index must be < size(). In debug builds (NDEBUG not defined) an
    // assertion fires on out-of-bounds access. In release builds the caller
    // is responsible for bounds checking — prefer get() for safe access.
    // Complexity: O(1).
    T operator[](uint32_t index) const {
        assert(index < count_);
        uint32_t physical = (head_ + Capacity - count_ + index) % Capacity;
        return buffer_[physical];
    }

    // Safe bounds-checked accessor. Sets `out` to the element at logical
    // index and returns true. Returns false without modifying `out` if
    // index >= size(). Prefer this over operator[] in production code paths.
    // Complexity: O(1).
    bool get(uint32_t index, T& out) const {
        if (index >= count_) {
            return false;
        }
        uint32_t physical = (head_ + Capacity - count_ + index) % Capacity;
        out = buffer_[physical];
        return true;
    }

    // Returns the number of valid elements currently stored (0..Capacity).
    uint32_t size() const {
        return count_;
    }

    // Returns the maximum number of elements the buffer can hold.
    // This is the Capacity template parameter, not a runtime value.
    uint32_t capacity() const {
        return Capacity;
    }

    // Returns true when size() == Capacity, i.e. the next push() will
    // overwrite the oldest element.
    bool is_full() const {
        return count_ == Capacity;
    }

private:
    std::array<T, Capacity> buffer_{};  // zero-initialised; static storage
    uint32_t head_  = 0;                // next write slot (wraps via % Capacity)
    uint32_t count_ = 0;                // number of valid elements [0, Capacity]
};

} // namespace micromind
