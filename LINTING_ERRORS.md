# Code Quality Audit Report

**Generated**: 2026-03-26 22:00
**Target**: `/r/OpenSource Projects/MicroMind-Anomaly`
**Languages Detected**: C++14
**Files Analyzed**: 8 (4 headers, 3 implementation files, 1 stub)
**Linters Used**: None available (g++ and clang++ not present in PATH; manual analysis applied)
**Total Issues**: 17

## Executive Summary

| Severity | Count |
|----------|-------|
| Critical | 0     |
| High     | 3     |
| Medium   | 6     |
| Low      | 8     |

The core ring buffer and inference engine implementations are structurally sound. The most significant findings are: (1) a real undefined behaviour gap in `ring_buffer.h`'s `operator[]` whose comment misrepresents the failure mode, (2) `int` literal arguments to `push()` on `RingBuffer<int16_t, …>` that will fire `-Wconversion` + `-Werror` and break the build, and (3) the test targets are compiled without the project's own `-Wall -Wextra -Werror` flags, meaning warning regressions in tests go undetected. No secrets or dangerous patterns were found.

---

## High Priority Issues

### HIG-01: `operator[]` out-of-bounds access is UB, not a safe default return
**File**: `include/micromind/ring_buffer.h:46–52`
**Category**: Bug

The method comment states "Behaviour is undefined (returns default-constructed T) when `i >= size()`." That claim is false: when `index >= size()`, the computed `physical` slot is within `[0, Capacity)` arithmetically, but the slot has never been written with a valid element — the buffer stores garbage (zero-initialised bytes at construction, but arbitrary values after earlier pushes and overwrites). More dangerously, if `index >= Capacity` the `% Capacity` operation keeps `physical` in-bounds for `buffer_`, but the caller would be reading a slot whose content is semantically undefined. The documentation creates a false sense of safety. In a safety-critical system a caller that receives a "default-constructed" value rather than a fault indicator will silently continue with bad data.

There is no bounds guard at all. The comment also conflicts with the code: there is no default construction, only a raw array read.

**Problematic Code:**
```cpp
T operator[](uint32_t index) const {
    // Behaviour is undefined (returns default-constructed T) when i >= size().
    uint32_t physical = (head_ + Capacity - count_ + index) % Capacity;
    return buffer_[physical];
}
```

**Recommended Fix:**
Add a bounds assertion (a no-op in release builds via a project-level `MICROMIND_ASSERT` macro, or a hard trap on embedded targets), and correct the comment to accurately state the behaviour:
```cpp
T operator[](uint32_t index) const {
    // Precondition: index < size(). Violating this precondition reads
    // from an uninitialised or stale slot — behaviour is undefined.
    // Use MICROMIND_ASSERT or a debug trap to catch violations in testing.
    MICROMIND_ASSERT(index < count_);
    uint32_t physical = (head_ + Capacity - count_ + index) % Capacity;
    return buffer_[physical];
}
```
At minimum, correct the comment so it does not assert a guarantee the code does not provide.

---

### HIG-02: `push()` called with plain `int` literals on `RingBuffer<int16_t, …>` — breaks build under `-Wconversion -Werror`
**File**: `tests/test_ring_buffer.cpp:39,43,47,51,68,69,70,71,75,81,100,101,102,121,165,171,176`
**Category**: Type Error

The test file constructs `RingBuffer<int16_t, N>` in several tests then calls `rb.push(10)`, `rb.push(1)`, `rb.push(42)`, etc. with undecorated `int` literals. `push(T value)` takes `T = int16_t`. The compiler must implicitly narrow `int` → `int16_t` at the call site. Under `-Wconversion` this generates a warning; under `-Werror` it becomes a build failure. Although `-Wconversion` is not part of `-Wall` or `-Wextra`, `-Wsign-conversion` is enabled by `-Wextra` and applies here. Many versions of GCC and all versions of Clang also emit `-Wnarrowing` or `-Wconversion` for this pattern under `-Wextra`.

Additionally, the test executable is **not** built with the project's `micromind_flags` (see `tests/CMakeLists.txt`), so these warnings are currently suppressed. See HIG-03.

Representative problematic lines:
```cpp
// test_ring_buffer.cpp — Test 3 (PushToCapacity)
micromind::RingBuffer<int16_t, 4> rb;
rb.push(10);   // int → int16_t: implicit narrowing
rb.push(20);
rb.push(30);
rb.push(40);

// test_ring_buffer.cpp — Test 8 (SingleElementCapacity)
rb.push(42);   // int → int16_t
rb.push(99);
rb.push(7);
```

**Recommended Fix:**
Use `int16_t` literals via explicit cast or the `static_cast` pattern already used in the `EXPECT_EQ` calls in the same file:
```cpp
rb.push(static_cast<int16_t>(10));
rb.push(static_cast<int16_t>(20));
rb.push(static_cast<int16_t>(30));
rb.push(static_cast<int16_t>(40));
```
Alternatively, declare typed local variables:
```cpp
constexpr int16_t v1 = 10, v2 = 20, v3 = 30, v4 = 40;
rb.push(v1); rb.push(v2); rb.push(v3); rb.push(v4);
```
Apply the same fix to all `push()` calls in `Test 4` (lines 68–71, 75, 81), `Test 5` (lines 100–102), and `Test 8` (lines 165, 171, 176).

---

### HIG-03: Test targets compiled without project warning flags — warning regressions in tests are invisible
**File**: `tests/CMakeLists.txt:1–16`
**Category**: Bug

`micromind_flags` (which carries `-Wall -Wextra -Werror -fno-exceptions`) is linked `PRIVATE` to the `micromind` library target only. The `micromind_tests` executable links `micromind` and `GTest::gtest_main` but **never** links `micromind_flags`. As a result, all test source files are compiled with the toolchain's default flags. This means:

- The `int` → `int16_t` implicit narrowing in HIG-02 produces no error at build time.
- Future test code can introduce any warning-generating pattern without CI catching it.
- The `-fno-exceptions` constraint is not enforced in tests, so test code could accidentally use exception-dependent constructs that would silently fail on the actual embedded target.

**Problematic Code:**
```cmake
# tests/CMakeLists.txt
add_executable(micromind_tests
    test_ring_buffer.cpp
    test_inference.cpp
)

target_link_libraries(micromind_tests
    PRIVATE
        micromind
        GTest::gtest_main   # <-- micromind_flags never linked here
)
```

**Recommended Fix:**
Link `micromind_flags` to the test executable. To avoid applying `-fno-exceptions` to GoogleTest itself (which requires exceptions), create a separate flags library for the warning-only subset:
```cmake
# In root CMakeLists.txt, alongside micromind_flags:
add_library(micromind_warn_flags INTERFACE)
if(NOT MSVC)
    target_compile_options(micromind_warn_flags INTERFACE
        -Wall -Wextra -Werror
        # Intentionally omits -fno-exceptions so GTest can use them
    )
endif()

# In tests/CMakeLists.txt:
target_link_libraries(micromind_tests
    PRIVATE
        micromind
        GTest::gtest_main
        micromind_warn_flags   # <-- add this
)
```

---

## Medium Priority Issues

### MED-01: `1u` template argument type is `unsigned int`, not `uint32_t` — portability hazard on LLP64 targets
**File**: `src/detector.cpp:59–60`
**Category**: Type Error

`mat_vec_mul` and `relu` are declared with `uint32_t` non-type template parameters. In C++14, a non-type template argument undergoes an implicit integral conversion to match the parameter type. The literal `1u` has type `unsigned int`. On LP64 and ILP32 platforms `unsigned int` and `uint32_t` are the same underlying type, so no conversion occurs. On **LLP64** platforms (Windows 64-bit with MSVC or MinGW), `uint32_t` is `unsigned long` while `unsigned int` is a distinct 32-bit type. Although the numeric value fits and the code will function correctly, compilers in strict mode may emit a diagnostic about the type mismatch, and the `micromind_flags` INTERFACE library already omits the MSVC branch entirely (line 13 of root `CMakeLists.txt`), leaving portability gaps unaudited.

```cpp
// src/detector.cpp:59–60
mat_vec_mul<1u, INPUT_FEATURES>(weights, input, output);  // 1u is unsigned int
relu<1u>(output);                                          // same issue
```

**Recommended Fix:**
Use `static_cast<uint32_t>` or a named `constexpr uint32_t` constant (C++14 has no `uint32_t` literal suffix):
```cpp
constexpr uint32_t OUTPUT_FEATURES = 1u;  // define alongside INPUT_FEATURES in config.h
mat_vec_mul<OUTPUT_FEATURES, INPUT_FEATURES>(weights, input, output);
relu<OUTPUT_FEATURES>(output);
```
This also gives the output dimension a meaningful name, which aids maintainability when the network is expanded to multi-output.

---

### MED-02: No upper-bound guard on `Capacity` in `ring_buffer.h` — silent overflow in index formula for extreme values
**File**: `include/micromind/ring_buffer.h:28,50`
**Category**: Bug

The `static_assert` on line 28 guards only the lower bound (`Capacity > 0u`). The index formula at line 50 computes `head_ + Capacity - count_ + index`. Since `head_` is in `[0, Capacity-1]`, the intermediate value `head_ + Capacity` reaches a maximum of `2 * Capacity - 1`. For `Capacity > UINT32_MAX / 2` (i.e., `Capacity >= 2147483648u`), this addition wraps around, producing a completely wrong physical index with no diagnostic.

While no current instantiation uses such a large capacity, the lack of a guard means a future refactor or copy-paste of this template could silently corrupt data.

```cpp
// include/micromind/ring_buffer.h:28
static_assert(Capacity > 0u, "RingBuffer Capacity must be greater than zero");
// Missing: no upper-bound guard

// Line 50 — vulnerable expression:
uint32_t physical = (head_ + Capacity - count_ + index) % Capacity;
```

**Recommended Fix:**
```cpp
static_assert(Capacity > 0u,       "RingBuffer Capacity must be greater than zero");
static_assert(Capacity <= 65536u,  "RingBuffer Capacity exceeds safe index arithmetic range for uint32_t");
```
The specific upper bound should match the largest realistic embedded use. `65536u` is a conservative but generous limit for any sensor window.

---

### MED-03: Synchronous callback creates re-entrancy risk in `push_sensor_value`
**File**: `src/detector.cpp:64–66` / `include/micromind/detector.h:33`
**Category**: Bug

`push_sensor_value()` fires the registered `StopCallback` synchronously before returning. The callback is an arbitrary `void(*)()`. If a callback implementation calls `push_sensor_value()` again (e.g., to record a diagnostic sample), this creates unbounded recursion. On an embedded system with a fixed stack (often 2–4 KB on Cortex-M0), stack overflow is a hard fault. There is no documentation warning callers against re-entrant use, and no guard in the implementation.

```cpp
// src/detector.cpp:64–66
if (callback_ != nullptr && output[0] > ANOMALY_THRESHOLD) {
    callback_();   // re-entrant call to push_sensor_value() from here → stack overflow
}
```

**Recommended Fix:**
Add a re-entrancy guard using a member flag:
```cpp
// detector.h — add to private section:
bool in_callback_ = false;

// detector.cpp:
if (callback_ != nullptr && output[0] > ANOMALY_THRESHOLD) {
    if (!in_callback_) {
        in_callback_ = true;
        callback_();
        in_callback_ = false;
    }
}
```
Also document the single-threaded, non-re-entrant contract explicitly in `detector.h`'s class comment — it currently says "Thread safety: none" but does not mention re-entrancy.

---

### MED-04: Missing test coverage for `operator[]` with out-of-bounds index
**File**: `tests/test_ring_buffer.cpp`
**Category**: Bug (test gap)

There is no test that exercises `operator[]` with `index >= size()`. The comment in `ring_buffer.h` promises a specific behaviour ("returns default-constructed T") that the code does not implement (see HIG-01). This divergence between documented contract and actual behaviour has survived because no test verifies the boundary. In a safety-critical library, every documented contract must have a corresponding test.

**Recommended Fix:**
Add a death test (if `MICROMIND_ASSERT` is wired to abort) or a behaviour test documenting the actual return:
```cpp
TEST(RingBuffer, OutOfBoundsIndexContractIsDocumented) {
    micromind::RingBuffer<int16_t, 4> rb;
    rb.push(static_cast<int16_t>(1));
    rb.push(static_cast<int16_t>(2));
    // rb[2] is out-of-bounds (size == 2). Either assert fires, or
    // document exactly which slot is read. Update this test when
    // HIG-01 is resolved to confirm the chosen contract.
    //
    // With MICROMIND_ASSERT enabled:
    // EXPECT_DEATH(rb[2], "");
}
```

---

### MED-05: No test for anomaly detection threshold boundary in `Detector`
**File**: `tests/` (missing file)
**Category**: Bug (test gap)

`Detector::push_sensor_value` fires the callback when `output[0] > ANOMALY_THRESHOLD`. There are no tests for `Detector` at all — not in `test_inference.cpp` (which only tests `mat_vec_mul` and `relu`) and not in any other file. Critical logic paths with no test coverage in a safety system represent a direct reliability risk:

- No test verifies that the callback fires exactly once per anomaly trigger (not multiple times per call).
- No test verifies the callback does NOT fire when output equals `ANOMALY_THRESHOLD` exactly (the `>` vs `>=` boundary).
- No test verifies zero-padding behaviour when the buffer has fewer than `INPUT_FEATURES` samples.
- No test verifies that `register_callback(nullptr)` safely clears a previously set callback.

**Recommended Fix:**
Add `tests/test_detector.cpp` with at minimum:
```cpp
TEST(Detector, CallbackNotFiredBelowThreshold) { /* all-zero inputs */ }
TEST(Detector, CallbackFiredAboveThreshold)    { /* all-one inputs with known weights */ }
TEST(Detector, CallbackNotFiredAtExactThreshold) { /* output == ANOMALY_THRESHOLD */ }
TEST(Detector, NullCallbackClearsRegistration)  { /* register then nullptr */ }
TEST(Detector, ZeroPaddingWhenBufferPartiallyFilled) { /* fewer than INPUT_FEATURES pushes */ }
```

---

### MED-06: `mat_vec_mul` and `relu` lack `noexcept` — contradicts `-fno-exceptions` mandate
**File**: `include/micromind/inference_engine.h:22–32, 45–50`
**Category**: Code Smell

The project is built with `-fno-exceptions`. Functions that cannot throw should be declared `noexcept` to: (a) document the no-throw contract explicitly, (b) enable compiler optimisations that eliminate exception-handling prologue/epilogue even on targets where the flag is not set (e.g., a future host-side test build), and (c) cause a compile-time error if an exception-throwing expression is accidentally introduced.

```cpp
// Current:
template<uint32_t Rows, uint32_t Cols>
void mat_vec_mul(const float (&matrix)[Rows][Cols],
                 const std::array<float, Cols>& vec,
                 std::array<float, Rows>& out)

template<uint32_t N>
void relu(std::array<float, N>& arr)
```

**Recommended Fix:**
```cpp
template<uint32_t Rows, uint32_t Cols>
void mat_vec_mul(const float (&matrix)[Rows][Cols],
                 const std::array<float, Cols>& vec,
                 std::array<float, Rows>& out) noexcept

template<uint32_t N>
void relu(std::array<float, N>& arr) noexcept
```
Apply the same to `push()`, `operator[]`, `size()`, `capacity()`, `is_full()` in `ring_buffer.h`, and to `register_callback()` / `push_sensor_value()` in `detector.h`.

---

## Low Priority Issues

### LOW-01: `static constexpr` at namespace scope is redundant — use `inline constexpr` or document rationale
**File**: `include/micromind/config.h:5–7`
**Category**: Style/Formatting

In C++14, `static constexpr` at namespace scope gives the constants internal linkage. The `static` keyword is doing useful work here (preventing ODR conflicts across TUs), but it results in each translation unit that includes `config.h` having its own copy of each constant. This wastes a small amount of `.rodata` space per TU on embedded targets. In C++17+ the correct idiom is `inline constexpr`. For this C++14 codebase `static constexpr` is acceptable, but the rationale should be stated in a comment to prevent a future maintainer from removing `static` thinking it is redundant (which would introduce ODR violations).

```cpp
// config.h:5–7 — no comment explaining why static is intentional
static constexpr uint32_t RING_BUFFER_CAPACITY = 32;
static constexpr float    ANOMALY_THRESHOLD     = 0.5f;
static constexpr uint32_t INPUT_FEATURES        = 4;
```

**Recommended Fix:**
```cpp
// static gives internal linkage in C++14, preventing ODR violations when
// this header is included in multiple translation units.
// In C++17+ these would be 'inline constexpr'.
static constexpr uint32_t RING_BUFFER_CAPACITY = 32;
static constexpr float    ANOMALY_THRESHOLD     = 0.5f;
static constexpr uint32_t INPUT_FEATURES        = 4;
```

---

### LOW-02: No upper-bound on `Capacity` for `std::array<T, Capacity>` stack allocation
**File**: `include/micromind/ring_buffer.h:72`
**Category**: Code Smell

`std::array<T, Capacity> buffer_` is allocated inline in the object. For large `Capacity` or large `T`, this can exhaust the stack if a `RingBuffer` is declared as a local variable. There is no documentation note or `static_assert` warning that `RingBuffer` instances should be declared at file or class scope (static storage) on stack-constrained targets.

**Recommended Fix:**
Add a comment to the class declaration:
```cpp
// NOTE: RingBuffer instances are large (sizeof(T) * Capacity bytes + 8).
// On embedded targets with constrained stacks, always declare at static
// or class scope, never as a local variable in a deep call stack.
```

---

### LOW-03: `weights` array in `push_sensor_value` is magic-number filled — no name or unit documentation
**File**: `src/detector.cpp:54–56`
**Category**: Code Smell

The placeholder weight `0.25f` is explained in a comment as yielding `max output = 1.0f` with 4 inputs. But the comment is inside the function body, not associated with any named constant. When real weights are substituted at calibration time there is no schema, unit, or version identifier. For a safety-critical system, weight provenance must be traceable.

```cpp
static const float weights[1][INPUT_FEATURES] = {
    { 0.25f, 0.25f, 0.25f, 0.25f }
};
```

**Recommended Fix:**
At minimum add a version identifier and calibration date:
```cpp
// Weights v0.0 — PLACEHOLDER. Replace with calibration output before release.
// Source: none (uniform dummy weights for integration testing only).
// Expected input range: [0.0, 1.0] per feature. Output range: [0.0, 1.0].
static const float weights[1][INPUT_FEATURES] = {
    { 0.25f, 0.25f, 0.25f, 0.25f }
};
```

---

### LOW-04: `inference_engine.cpp` is an empty stub — could cause confusion at link time
**File**: `src/inference_engine.cpp:1–2`
**Category**: Code Smell

The file contains only `#include "micromind/inference_engine.h"`. Since all of `inference_engine.h`'s functions are templates defined inline, this `.cpp` stub is a no-op. It is listed in `src/CMakeLists.txt` as a source file of the `micromind` library. While harmless, it adds an unnecessary compilation unit and will confuse maintainers who expect a `.cpp` to contain definitions.

**Recommended Fix:**
Either delete `inference_engine.cpp` and remove it from `src/CMakeLists.txt`, or add a comment at the top:
```cpp
// inference_engine.cpp — intentional stub.
// All inference_engine.h functions are templates defined in the header.
// This file is included in the build system only to reserve a conventional
// place for any future non-template helpers.
#include "micromind/inference_engine.h"
```

---

### LOW-05: `test_inference.cpp` lacks template-argument-deduction failure tests for dimension mismatches
**File**: `tests/test_inference.cpp`
**Category**: Code Smell

All four `mat_vec_mul` tests use square or explicitly matched dimensions. There is no test that verifies a compile-time error when `vec.size()` does not match `Cols`, or `out.size()` does not match `Rows`. These errors are caught at compile time by the template system, not at runtime, so they cannot be tested as runtime EXPECT_* assertions — but they should be documented in a negative-compilation test list (e.g., via a commented-out snippet marked `// should not compile`) or in a separate `static_assert`-based test to confirm the type system is working as intended.

---

### LOW-06: `RingBuffer::push` does not document the "lossy overwrite" semantic in the header API comment
**File**: `include/micromind/ring_buffer.h:30–41`
**Category**: Code Smell

The class-level comment (lines 7–24) documents the lossy overwrite. The `push()` method comment (line 31) also mentions it. However the method signature itself has no `[[deprecated]]` marker, `[[nodiscard]]` consideration, or return value to indicate that data was lost. A caller cannot distinguish between a `push()` that stored a new unique value and one that silently discarded data. In a sensor anomaly detector this distinction matters — a burst of samples during a high-frequency event could silently drop readings.

**Recommended Fix:**
Consider returning a `bool` from `push()` indicating whether an overwrite occurred, or adding a `bool overflow_pending_` flag readable via a separate accessor. At minimum, document clearly in the method comment that no notification is given on overwrite:
```cpp
// Pushes value into the buffer. Returns true if an element was silently
// overwritten (i.e., the buffer was already full). Returns false otherwise.
bool push(T value);
```

---

### LOW-07: Test 6 loop variable type could trigger `-Wsign-compare` on some compilers
**File**: `tests/test_ring_buffer.cpp:121`
**Category**: Type Error

```cpp
for (int16_t i = 1; i <= 8; ++i) {
    rb.push(i);
}
```

The loop bound `8` is an `int` literal. The comparison `i <= 8` promotes `int16_t i` to `int`, then compares two `int` values — no warning results. However `++i` on `int16_t` returns `int16_t`, which is correct. The `rb.push(i)` call is safe here because `i` is already `int16_t`. This loop is the one correctly-typed case in the test file and is worth keeping as-is. No action required — noting for completeness.

---

### LOW-08: `register_callback` and `push_sensor_value` in `detector.h` are not marked `noexcept`
**File**: `include/micromind/detector.h:27,33`
**Category**: Code Smell

Same rationale as MED-06. Both public methods of `Detector` have no failure mode and should be `noexcept` to document the contract and allow the compiler to optimise call sites.

```cpp
void register_callback(StopCallback cb);       // should be noexcept
void push_sensor_value(float value);           // should be noexcept
```

**Recommended Fix:**
```cpp
void register_callback(StopCallback cb) noexcept;
void push_sensor_value(float value) noexcept;
```

---

## Summary by File

| File | Critical | High | Medium | Low | Total |
|------|----------|------|--------|-----|-------|
| `include/micromind/ring_buffer.h` | 0 | 1 | 1 | 2 | 4 |
| `include/micromind/inference_engine.h` | 0 | 0 | 1 | 1 | 2 |
| `include/micromind/detector.h` | 0 | 0 | 1 | 1 | 2 |
| `include/micromind/config.h` | 0 | 0 | 0 | 1 | 1 |
| `src/detector.cpp` | 0 | 0 | 2 | 1 | 3 |
| `src/inference_engine.cpp` | 0 | 0 | 0 | 1 | 1 |
| `tests/test_ring_buffer.cpp` | 0 | 1 | 1 | 1 | 3 |
| `tests/test_inference.cpp` | 0 | 0 | 0 | 1 | 1 |
| `tests/CMakeLists.txt` | 0 | 1 | 0 | 0 | 1 |
| `tests/` *(missing detector tests)* | 0 | 0 | 1 | 0 | 1 |
| **Total** | **0** | **3** | **6** | **8** | **17** |

## Summary by Category

| Category | Count |
|----------|-------|
| Bug | 5 |
| Type Error | 3 |
| Code Smell | 7 |
| Resource Leak | 0 |
| Security | 0 |
| Style/Formatting | 2 |
