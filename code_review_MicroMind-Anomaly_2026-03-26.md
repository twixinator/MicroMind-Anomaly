# Code Review: MicroMind-Anomaly
Date: 2026-03-26 | Files reviewed: 11 | Lines: ~350

## Executive Summary
The library is well-structured for an embedded C++14 target: no heap allocation, no exceptions in
production code, and all storage is static. Two issues require attention before shipping: the
`-fno-exceptions` flag propagates transitively to the test binary (which uses GoogleTest, a
framework that requires exceptions), and `RingBuffer<T, 0>` causes integer division by zero at
runtime with no compile-time guard. All other findings are documentation inaccuracies or
low-severity hygiene items.

Critical: 0 | High: 1 | Medium: 2 | Low: 2 | Risk: **High**

---

## High Findings

### [HIGH-001] `-fno-exceptions` propagates transitively to GoogleTest test sources
**File:** `src/CMakeLists.txt:11` and `tests/CMakeLists.txt:8-12`

**Problem:**
`micromind` links `micromind_flags` as `PUBLIC`. Under CMake's transitive dependency model,
`PUBLIC` INTERFACE compile options on a library are forwarded to every consumer of that library —
including consumers that link it `PRIVATE`. When `micromind_tests` links `micromind` (even
PRIVATELY), CMake applies `micromind_flags`' `-fno-exceptions` to the compilation of
`test_ring_buffer.cpp` and `test_inference.cpp`. GoogleTest itself is compiled separately (via
FetchContent) and does not receive this flag, but the test source files that `#include
<gtest/gtest.h>` and call `ASSERT_*` macros are compiled with `-fno-exceptions`. On GCC/Clang,
GoogleTest's ASSERT macros use a `return` trick rather than exceptions to abort, so this often
works in practice — but GoogleTest's death-test infrastructure and some internal helpers do rely
on exceptions. More importantly this is a correctness contract violation: the project's own rules
state `-fno-exceptions` must not apply to GoogleTest. This is also a latent ABI mismatch: if any
GTest header instantiates templates that differ between exception and no-exception modes, mixing
the two compilation units is undefined behavior.

**Evidence:**
```cmake
# src/CMakeLists.txt line 10-12: PUBLIC propagation
target_link_libraries(micromind
    PUBLIC micromind_flags    # <-- PUBLIC means consumers inherit -fno-exceptions
)
```
```cmake
# tests/CMakeLists.txt lines 8-12: test binary is a consumer
target_link_libraries(micromind_tests
    PRIVATE
        micromind             # <-- inherits micromind_flags transitively
        GTest::gtest_main
)
```

**Fix:**
Change `micromind_flags` from `PUBLIC` to `PRIVATE` in `src/CMakeLists.txt`. The flags only need
to govern how `micromind`'s own source files are compiled, not how consumers compile their code.
The include path (`target_include_directories ... PUBLIC`) should remain PUBLIC so consumers can
still find the headers.

```cmake
# src/CMakeLists.txt — corrected
add_library(micromind STATIC
    inference_engine.cpp
    detector.cpp
)

target_include_directories(micromind
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(micromind
    PRIVATE micromind_flags   # flags govern only micromind's own compilation units
)
```

If the test executables themselves also need `-Wall -Wextra -Werror` (but NOT `-fno-exceptions`),
create a separate interface library for that purpose and apply it directly to `micromind_tests`.

**Effort:** 1 line change in `src/CMakeLists.txt`.

---

## Medium Findings

### [MED-001] `RingBuffer<T, 0>` instantiation causes division by zero with no compile-time guard
**File:** `include/micromind/ring_buffer.h:34` and `:49`

**Problem:**
The `Capacity` template parameter is `uint32_t`. Nothing prevents instantiation with `Capacity =
0`. Two operations then produce integer division by zero (undefined behavior):
- `push()` line 34: `head_ = static_cast<uint32_t>((head_ + 1u) % Capacity)` — `% 0` is UB.
- `operator[]` line 49: `(head_ + Capacity - count_ + index) % Capacity` — `% 0` is UB.

On a bare-metal embedded target with no MMU this will likely cause a hard fault. Because
`Capacity` is a compile-time constant, the compiler can and will emit a constant `% 0` that some
compilers (GCC -O2) will turn into a trap instruction at compile time, while others silently
generate a divide instruction that faults at runtime.

**Evidence:**
```cpp
// ring_buffer.h:34
head_ = static_cast<uint32_t>((head_ + 1u) % Capacity);  // UB when Capacity==0

// ring_buffer.h:49
uint32_t physical = (head_ + Capacity - count_ + index) % Capacity;  // UB when Capacity==0
```

**Fix:**
Add a `static_assert` at the top of the class definition:

```cpp
template<typename T, uint32_t Capacity>
class RingBuffer {
    static_assert(Capacity > 0u, "RingBuffer Capacity must be greater than zero");
    // ... rest of class unchanged
};
```

This converts the silent runtime fault into a clear compile-time error with a readable message.

**Effort:** 1 line.

---

### [MED-002] `operator[]` documentation contradicts actual behavior (silent wrong-slot read on out-of-bounds)
**File:** `include/micromind/ring_buffer.h:43-44`

**Problem:**
The comment states: *"Behaviour is undefined (returns default-constructed T) when i >= size()."*
This is incorrect on both counts:
1. The behavior is not "undefined" in the C++ language sense — `% Capacity` keeps the physical
   index within `[0, Capacity)`, so no out-of-bounds memory access occurs.
2. The function does not return a default-constructed T. It returns whatever value currently
   occupies the wrapped physical slot, which is an arbitrary older sample. A caller that passes
   `index = size()` silently receives stale sensor data instead of an error indicator, which is a
   correctness hazard in safety-critical anomaly detection.

The real behavior is: on out-of-bounds access, a valid but logically incorrect element is
returned. There is no runtime trap, no sentinel value, and no error return.

**Evidence:**
```cpp
// ring_buffer.h:43-44 — comment is wrong
// Behaviour is undefined (returns default-constructed T) when i >= size().

T operator[](uint32_t index) const {
    // % Capacity keeps physical in bounds; no default-construct path exists
    uint32_t physical = (head_ + Capacity - count_ + index) % Capacity;
    return buffer_[physical];   // returns an arbitrary existing element
}
```

**Fix — option A (preferred for safety-critical code):** Return a sentinel / use an output
parameter with an error code, consistent with the no-exceptions rule:

```cpp
// Returns false and leaves *out untouched if index is out of range.
bool get(uint32_t index, T& out) const {
    if (index >= count_) {
        return false;
    }
    uint32_t physical = (head_ + Capacity - count_ + index) % Capacity;
    out = buffer_[physical];
    return true;
}
```

**Fix — option B (minimal change):** At minimum, correct the comment and add an assertion for
debug builds:

```cpp
// Behaviour is UNSPECIFIED (an arbitrary existing slot is returned) when
// index >= size(). In debug builds the assertion fires; in release builds
// the caller is responsible for bounds checking.
T operator[](uint32_t index) const {
    // Debug guard — does not fire in -DNDEBUG release builds
    assert(index < count_);
    uint32_t physical = (head_ + Capacity - count_ + index) % Capacity;
    return buffer_[physical];
}
```

**Effort:** Option B is 2 lines. Option A requires updating callers in `detector.cpp`.

---

## Low Findings

### [LOW-001] `inference_engine.cpp` is an empty translation unit
**File:** `src/inference_engine.cpp:1`

**Problem:**
The file contains only `#include "micromind/inference_engine.h"`. All functions in
`inference_engine.h` are function templates defined entirely in the header. The `.cpp` file
contributes zero object code and exists solely as a compilation smoke-test (verifying the header
includes compile cleanly in isolation). This is harmless but adds a spurious object file to the
archive and may confuse future maintainers who expect it to contain definitions.

**Evidence:**
```cpp
// src/inference_engine.cpp — entire file
#include "micromind/inference_engine.h"
```

**Fix:**
Either delete the file and remove it from `src/CMakeLists.txt`, or add a comment explaining its
purpose:

```cpp
// inference_engine.cpp
// This translation unit is intentionally empty.
// All inference engine functions are function templates defined in
// inference_engine.h and instantiated at the call site in detector.cpp.
// This file exists only to verify the header compiles in isolation.
#include "micromind/inference_engine.h"
```

**Effort:** 3 lines of comment, or delete the file.

---

### [LOW-002] `config.h` uses `static constexpr` at namespace scope (correct for C++14, but worth noting)
**File:** `include/micromind/config.h:5-8`

**Problem:**
In C++14, `static constexpr` at namespace scope gives each translation unit its own copy of the
constant (internal linkage). This is the correct and required pattern in C++14 — it is not a bug.
However, if the project ever moves to C++17, these should become `inline constexpr` (without
`static`) to have external linkage and a single definition across TUs. The current form also
produces a `-Wunused-variable` warning in some configurations when a TU includes the header but
never references a particular constant.

**Evidence:**
```cpp
// config.h:5-8
static constexpr uint32_t RING_BUFFER_CAPACITY = 32;
static constexpr float    ANOMALY_THRESHOLD     = 0.5f;
static constexpr uint32_t INPUT_FEATURES        = 4;
static constexpr uint32_t HIDDEN_UNITS          = 8;
```

`HIDDEN_UNITS` is defined but never referenced in any reviewed source file — it is dead code that
may trigger `-Wunused-variable` under `-Wextra` in some TUs.

**Fix:**
Remove `HIDDEN_UNITS` if it is not used, or annotate its intended use. If C++17 is ever targeted:

```cpp
// C++17 form (do NOT apply yet — project is C++14):
// inline constexpr uint32_t HIDDEN_UNITS = 8;
```

For now, remove the unused constant:

```cpp
// config.h — remove or comment out the unused constant
// static constexpr uint32_t HIDDEN_UNITS = 8;  // unused — remove until needed
```

**Effort:** Delete 1 line.

---

## Security Checklist

| Category | Status |
|---|---|
| Dynamic memory (malloc/new/vector) | OK — none found in any source file |
| Exception usage (throw/try/catch) | OK — none found in production code |
| C++17+ features | OK — no `if constexpr`, `std::optional`, `std::variant`, or `[[nodiscard]]` found |
| External ML dependencies | OK — none; all arithmetic is self-contained |
| Callback null-check before invoke | OK — `detector.cpp:64` checks `callback_ != nullptr` before calling |
| Ring buffer index bounds (physical) | OK — `% Capacity` keeps physical index within array bounds at all times |
| Ring buffer logical bounds | Issue — see MED-002 (silent wrong-slot read, no error path) |
| Static storage only | OK — `std::array` members, function-local static const weights |
| `-fno-exceptions` flag scoping | Issue — see HIGH-001 (propagates to test binary via PUBLIC linkage) |
| `Capacity == 0` guard | Issue — see MED-001 (no static_assert, division by zero at runtime) |
| Signed/unsigned comparison warnings | OK — loop variables and comparisons consistently use `uint32_t` |
| Exact `<cstdint>` types | OK — `int16_t`, `uint32_t`, `float` used throughout |
| CMake `add_compile_options` / `CMAKE_CXX_FLAGS` | OK — neither global mechanism is used; flags are INTERFACE-scoped |

---

## Priority Action Plan

1. **Fix HIGH-001 (1-line CMake change, must fix before CI is reliable):** Change
   `PUBLIC micromind_flags` to `PRIVATE micromind_flags` in `src/CMakeLists.txt`. Until this is
   fixed, `-fno-exceptions` is silently applied to the GoogleTest test sources, violating the
   stated project rule and risking ABI mismatches with the GTest runtime library.

2. **Fix MED-001 (1-line static_assert):** Add `static_assert(Capacity > 0u, ...)` to
   `RingBuffer`. The cost is one line; the benefit is a hard compile-time failure instead of a
   silent embedded hard fault.

3. **Fix MED-002 (correct comment + add assert):** The `operator[]` out-of-bounds comment is
   actively wrong and describes behavior that does not exist. At minimum correct the comment and
   add a debug `assert`. For safety-critical production use, add a bounds-checked `get()` method
   and migrate callers in `detector.cpp` to use it — this ensures the anomaly detector never
   silently feeds stale sensor data to the inference engine.

4. **LOW-002 (remove `HIDDEN_UNITS`):** The constant is defined but unreferenced. Under
   `-Wextra -Werror`, GCC may emit an `unused-variable` warning for TUs that include `config.h`
   without referencing it, turning a build warning into a build failure. Remove it or mark it
   with a comment indicating it is reserved for a future hidden layer.

5. **LOW-001 (document empty TU):** Add a comment to `inference_engine.cpp` explaining why it
   exists, or delete it. This prevents future contributors from adding misplaced definitions into
   a file that looks like it should have content.
