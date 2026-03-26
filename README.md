# MicroMind-Anomaly — Minimal edge AI anomaly detection for embedded C++

**Language:** C++14 | **License:** TBD | **Platform:** STM32 / ESP32 | **Build:** host (GCC 15 / MSYS2)

---

## What It Does

MicroMind-Anomaly is a bare-metal C++ library that performs local anomaly detection on sensor streams
typical of 3D printer systems — hotend temperature, bed temperature, extruder motor current, and
similar continuous signals. All inference runs on-device: no network, no cloud dependency, no RTOS
requirement. When the inference output exceeds the configured threshold, a registered `StopCallback`
is called synchronously, allowing the host firmware to halt the print or cut power before damage
occurs. The entire library uses static allocation only — no heap, no exceptions — making it
suitable for safety-critical bare-metal deployments where `malloc` failure or stack unwinding are
unacceptable.

---

## Key Constraints

- No `malloc`, `new`, or `std::vector` at runtime — all storage is statically allocated
- No exceptions (`-fno-exceptions` enforced project-wide via CMake)
- C++14 standard, compatible with GCC-ARM toolchain (`arm-none-eabi-g++`)
- All integer types sourced from `<cstdint>`; no implicit `int` widening
- `StopCallback` is a raw function pointer (`void(*)()`) — no `std::function`, no heap-based type erasure
- Single-threaded only — no internal synchronisation primitives

---

## Architecture

```
                           +---------------------+
  float                    |                     |
 SensorValue ─────push()──>│   RingBuffer<T, N>  │
                            |  (sliding window)   |
                            +─────────────────────+
                                      │
                              extract last
                              INPUT_FEATURES
                              samples (zero-pad
                              if buffer not full)
                                      │
                                      ▼
                            std::array<float, 4>  (input vector)
                                      │
                               mat_vec_mul()
                              (dense layer fwd pass)
                                      │
                                      ▼
                            std::array<float, 1>  (raw output)
                                      │
                                  relu()
                                      │
                                      ▼
                              output[0] > ANOMALY_THRESHOLD ?
                                      │
                                Yes ──┴──> StopCallback()
```

Linear data flow: `SensorValue → RingBuffer → Detector → mat_vec_mul → relu → threshold check → StopCallback`

---

## API Overview

### `RingBuffer<T, N>` — `include/micromind/ring_buffer.h`

Fixed-capacity circular buffer with static storage. Logical index 0 is always the oldest element,
matching the expected input layout for the downstream inference engine.

| Method | Description |
|--------|-------------|
| `void push(T value)` | Appends a value. Silently overwrites the oldest element when full. O(1). |
| `T operator[](uint32_t index) const` | Returns element at logical index `i` (0 = oldest). O(1). |
| `uint32_t size() const` | Number of valid elements currently held (0..Capacity). |
| `uint32_t capacity() const` | Returns the `Capacity` template parameter. |
| `bool is_full() const` | True when `size() == Capacity`. |

When full, `push()` advances `head_` and overwrites the oldest slot; `size()` remains pinned at
`Capacity`. This lossy sliding-window semantic is intentional for continuous sensor streams.

### `Detector` — `include/micromind/detector.h`

Accumulates samples and runs a single-layer forward pass on each new value.

| Method | Description |
|--------|-------------|
| `void register_callback(StopCallback cb)` | Registers the stop callback. Pass `nullptr` to clear. |
| `void push_sensor_value(float value)` | Pushes a sample, runs inference, fires callback if threshold exceeded. |

`push_sensor_value` calls `callback_` synchronously before returning. The null-check on
`callback_` is deliberate — on targets without an MMU a branch-to-null causes an immediate
hard fault, which is worse than a missed callback.

### `config.h` — `include/micromind/config.h`

All tunable constants are `static constexpr` values in `namespace micromind`:

| Constant | Default | Description |
|----------|---------|-------------|
| `RING_BUFFER_CAPACITY` | `32` | Sliding window depth |
| `ANOMALY_THRESHOLD` | `0.5f` | Inference output level that triggers the callback |
| `INPUT_FEATURES` | `4` | Number of input features fed to the dense layer |
| `OUTPUT_FEATURES` | `1` | Number of output neurons |

### `inference_engine.h` — `include/micromind/inference_engine.h`

Header-only primitives used inside `Detector::push_sensor_value`:

- `mat_vec_mul<Rows, Cols>(matrix, vec, out)` — dense layer forward pass. Zeroes `out` before
  accumulation; callers do not need to pre-initialise it.
- `relu<N>(arr)` — in-place ReLU activation. Uses `std::max(0.0f, x)` to allow the compiler to
  emit branchless `MAXSS`/`VMAXPS` on x86 and equivalent on Cortex-M4/M7 with FPU.

---

## Build (Host)

The host build targets MSYS2 UCRT64 with GCC and CMake installed. GoogleTest v1.14.0 is fetched
automatically via `FetchContent` at configure time — an internet connection is required on first
build.

```bash
# From an MSYS2 ucrt64 shell:
cmake -S . -B build -G Ninja
cmake --build build
./build/tests/micromind_tests.exe   # 15 tests
```

Compiler flags applied to all project targets (GoogleTest is excluded):
`-Wall -Wextra -Werror -fno-exceptions`

Cross-compilation for STM32/ESP32 is not yet wired into the build system — see v0.8 in the
roadmap below.

---

## Project Status / Roadmap

| Version | Status | Theme |
|---------|--------|-------|
| v0.1 | Done | Core library (ring buffer, inference engine, detector) |
| v0.2 | In progress | Foundation hardening + Detector tests |
| v0.3 | Planned | Multi-layer inference (hidden layer) |
| v0.4 | Planned | Multi-sensor input + feature extraction |
| v0.5 | Planned | Weight loading pipeline |
| v0.6 | Planned | ML training pipeline |
| v0.7 | Planned | Anomaly scoring and hysteresis |
| v0.8 | Planned | STM32 / ESP32 hardware port |
| v1.0 | Planned | Production hardening + CI/CD |

The current weight values in `detector.cpp` (`0.25f` for all four inputs) are placeholder stubs.
With `INPUT_FEATURES=4` inputs all at full scale the raw output reaches `1.0f`, which clears
`ANOMALY_THRESHOLD=0.5f` and exercises the callback path in integration tests. Replace with
calibrated weights once the training pipeline (v0.6) is complete.

---

## License

See LICENSE file.
