# MicroMind-Anomaly — Roadmap & Next Steps

**Date:** 2026-03-26
**Scope:** Full project — embedded C++ library, ML training pipeline, hardware port, production hardening
**Structure:** Phase-gated with definition-of-done per phase
**Baseline:** v0.1 complete — ring buffer, single-layer inference, detector, 15 passing tests

---

## Current State (v0.1 baseline)

| Component | Status |
|-----------|--------|
| `RingBuffer<T, N>` | Complete, 8 tests |
| `mat_vec_mul` + `relu` | Complete, 7 tests |
| `Detector` (single-layer, scalar input) | Functional, zero tests |
| Weights | Placeholder `0.25f` uniform — not trained |
| Hidden layer (`HIDDEN_UNITS = 8`) | Defined in design, not implemented |
| Multi-sensor input | Not implemented |
| Input normalization | Not implemented |
| Training pipeline | Not implemented |
| Hardware port (STM32 / ESP32) | Not implemented |
| CI/CD | Not implemented |

**Open review items carried into v0.2:**
- `operator[]` comment incorrect; no OOB guard (MED-002)
- `inference_engine.cpp` is a hollow TU (LOW-001)
- CMake `DOWNLOAD_EXTRACT_TIMESTAMP` warning

---

## Phase v0.2 — Foundation Hardening

**Goal:** Zero open review issues, full Detector test coverage. No new features until the foundation is solid.

### Tasks

| # | File | Change |
|---|------|--------|
| 1 | `include/micromind/ring_buffer.h` | Fix `operator[]` comment (actual behaviour: wraps, does not return default-T); add `assert(index < count_)` in debug builds |
| 2 | `include/micromind/ring_buffer.h` | Add `bool get(uint32_t index, T& out) const` — bounds-checked accessor returning `false` on OOB; production paths use this instead of raw `operator[]` |
| 3 | `tests/test_detector.cpp` | New file: callback fires above threshold; callback does not fire below threshold; null callback is safe; zero-padding on cold start (fewer than `INPUT_FEATURES` samples) |
| 4 | `src/inference_engine.cpp` | Add comment explaining the file is a CMake anchor TU; all definitions are header-only templates |
| 5 | `CMakeLists.txt` | Add `DOWNLOAD_EXTRACT_TIMESTAMP TRUE` to `FetchContent_Declare` to suppress CMP0135 warning |

### Success Criteria
- All tests green, including ≥ 4 new `Detector` tests
- Zero CMake warnings on configure
- No open items from `code_review_MicroMind-Anomaly_2026-03-26.md` or `LINTING_ERRORS.md`

### Subagents
- **Implementation:** `senior-fullstack-dev`
- **Verification:** `pr-review-toolkit:code-reviewer`

---

## Phase v0.3 — Multi-layer Inference

**Goal:** Implement the planned but absent hidden layer; make the network architecture configurable at compile time via `config.h`.

### Tasks

| # | File | Change |
|---|------|--------|
| 1 | `include/micromind/inference_engine.h` | Add `dense_forward<Rows, Cols>`: `mat_vec_mul` + bias vector + activation in one call; signature: `dense_forward(weights, bias, input, output, activation_fn)` |
| 2 | `include/micromind/inference_engine.h` | Activation function type: `using ActivationFn = void(*)(std::array<float,...>&)` — but templates make this awkward; use a template parameter tag (`struct ReLUTag{}`) instead |
| 3 | `include/micromind/config.h` | Restore `HIDDEN_UNITS = 8`; add `LAYER1_BIAS_INIT = 0.0f`, `LAYER2_BIAS_INIT = 0.0f` |
| 4 | `src/detector.cpp` | Replace single-layer forward pass with two-layer: input→hidden (ReLU) → hidden→output; bias arrays as `static const` locals |
| 5 | `tests/test_inference.cpp` | Add tests: known-value two-layer forward pass; bias offset shifts output correctly; ReLU clamps at hidden layer |

### Success Criteria
- `HIDDEN_UNITS` is live and used in the forward pass
- `detector.cpp` uses two-layer path; no dead single-layer code remains
- All new and existing tests green

### Subagents
- **Implementation:** `senior-fullstack-dev`

---

## Phase v0.4 — Multi-sensor Input & Feature Extraction

**Goal:** Replace the scalar ring buffer with a multi-channel design matching real 3D printer sensor streams (temperature + motor current at minimum).

### Tasks

| # | File | Change |
|---|------|--------|
| 1 | `include/micromind/config.h` | Add `NUM_CHANNELS = 2` and `FEATURES_PER_CHANNEL = 2` (mean + variance per channel); `INPUT_FEATURES` becomes `NUM_CHANNELS * FEATURES_PER_CHANNEL = 4` — derived, not hardcoded |
| 2 | `include/micromind/sensor_frame.h` | New file: `struct SensorFrame { float temp; float current; }` — POD, no heap, `static_assert(sizeof(SensorFrame) == NUM_CHANNELS * sizeof(float))` |
| 3 | `include/micromind/feature_extractor.h` | New file: `FeatureExtractor<N>` — takes `RingBuffer<SensorFrame, N>`, computes per-channel mean and variance over the window → returns `std::array<float, INPUT_FEATURES>` |
| 4 | `include/micromind/normalizer.h` | New file: `Normalizer` — `static constexpr` per-channel mean/std baked at compile time from training data; `normalize(std::array<float, INPUT_FEATURES>&)` in-place |
| 5 | `include/micromind/detector.h` | Replace `push_sensor_value(float)` with `push_frame(SensorFrame)` |
| 6 | `src/detector.cpp` | Wire: `push_frame` → ring buffer → feature extractor → normalizer → two-layer inference → scorer |
| 7 | `tests/test_feature_extractor.cpp` | New file: known window → expected mean/variance; single-element window; full window |
| 8 | `tests/test_normalizer.cpp` | New file: zero-mean unit-variance output for matching input; clipping behaviour |

### Success Criteria
- `push_frame(SensorFrame)` is the only public entry point for sensor data
- `INPUT_FEATURES` is derived from `NUM_CHANNELS` and `FEATURES_PER_CHANNEL` — not hardcoded
- All new and existing tests green

### Subagents
- **Implementation:** `senior-fullstack-dev`

---

## Phase v0.5 — Weight Loading & Model Serialization

**Goal:** Replace hardcoded placeholder weights with a reproducible pipeline that compiles trained weights into the binary.

### Tasks

| # | File | Change |
|---|------|--------|
| 1 | `include/micromind/weights.h` | Define generated file format: `constexpr float LAYER1_WEIGHTS[HIDDEN_UNITS][INPUT_FEATURES]`, `LAYER1_BIAS[HIDDEN_UNITS]`, `LAYER2_WEIGHTS[OUTPUT_FEATURES][HIDDEN_UNITS]`, `LAYER2_BIAS[OUTPUT_FEATURES]` |
| 2 | `tools/export_weights.py` | Python script: reads trained model (NumPy `.npy` or PyTorch `.pt`), validates shapes against `config.h` constants, emits valid `weights.h` |
| 3 | `src/detector.cpp` | Replace inline weight literals with `#include "micromind/weights.h"` |
| 4 | `CMakeLists.txt` | Add CMake option `MICROMIND_WEIGHTS_FILE` to point at a custom weights header path |
| 5 | `include/micromind/weights.h` (generated) | Add `static_assert` checks: array dimensions match `INPUT_FEATURES`, `HIDDEN_UNITS`, `OUTPUT_FEATURES` |
| 6 | `tools/default_weights.h` | Checked-in default (uniform `0.0f` or identity-like) so the build works before training |

### Success Criteria
- `export_weights.py` → rebuild → ctest is the complete model update cycle
- Changing weights never requires editing any `.cpp` or `.h` file by hand
- `static_assert` catches shape mismatches at compile time

### Subagents
- **Export script:** `senior-python-dev`
- **C++ integration:** `senior-fullstack-dev`

---

## Phase v0.6 — ML Training Pipeline

**Goal:** Produce a fully trainable anomaly detector end-to-end: raw sensor logs → `weights.h` → working firmware.

### Tasks

| # | File | Change |
|---|------|--------|
| 1 | `tools/README_data_format.md` | Define CSV log format: `timestamp_ms, temp_c, current_a, label` (`label` optional: `0`=normal, `1`=anomaly) |
| 2 | `tools/train.py` | Training script: loads CSV, builds sliding-window dataset (`RING_BUFFER_CAPACITY` samples), trains shallow autoencoder (PyTorch default) or one-class SVM (scikit-learn, via `--model svm` flag) |
| 3 | `tools/train.py` | Threshold calibration: compute 99th-percentile reconstruction error on normal hold-out set → writes `ANOMALY_THRESHOLD` into a `config_trained.h` alongside `weights.h` |
| 4 | `tools/eval.py` | Evaluation: precision, recall, F1 on labeled test CSV; confusion matrix printed to stdout; optional `--plot` flag for matplotlib |
| 5 | `tools/generate_synthetic_data.py` | Synthetic data generator for development/CI: normal distribution for baseline, Gaussian spike injection for anomalies |
| 6 | `docs/training_pipeline.md` | End-to-end walkthrough: data collection → `train.py` → `export_weights.py` → `cmake --build` → `ctest` |
| 7 | `requirements.txt` | Pin: `torch`, `scikit-learn`, `numpy`, `pandas`, `matplotlib` |

### Success Criteria
- Running `python train.py data/normal.csv && python export_weights.py && cmake --build build && ctest` end-to-end produces a working detector
- `eval.py` reports F1 ≥ 0.85 on synthetic dataset
- Pipeline documented and reproducible

### Subagents
- **Implementation:** `senior-python-dev`

---

## Phase v0.7 — Anomaly Scoring & Hysteresis

**Goal:** Prevent false-positive machine stops from single noisy samples; add graduated severity levels.

### Tasks

| # | File | Change |
|---|------|--------|
| 1 | `include/micromind/config.h` | Add `HYSTERESIS_WINDOW = 3` (consecutive breaches to trigger stop), `EMA_ALPHA = 0.3f` (smoothing factor), `WARNING_THRESHOLD = 0.4f` |
| 2 | `include/micromind/anomaly_scorer.h` | New file: `AnomalyScorer` — static `RingBuffer<float, HYSTERESIS_WINDOW>` of recent scores; computes EMA; exposes `update(float score) -> SeverityLevel` |
| 3 | `include/micromind/anomaly_scorer.h` | `enum class SeverityLevel : uint8_t { Normal, Warning, Critical }` |
| 4 | `include/micromind/detector.h` | Two callback slots: `StopCallback warning_cb_` and `StopCallback stop_cb_`; `register_warning_callback(StopCallback)`, `register_stop_callback(StopCallback)` |
| 5 | `src/detector.cpp` | Route output through `AnomalyScorer`; fire `warning_cb_` on `Warning`, `stop_cb_` on `Critical` |
| 6 | `tests/test_anomaly_scorer.cpp` | New file: EMA convergence with known inputs; hysteresis fires only after N consecutive triggers; single spike does not fire stop; warning fires before stop |

### Success Criteria
- A single-sample spike never fires the stop callback
- N consecutive threshold breaches always fire the stop callback
- Warning fires before stop at a lower threshold
- All new and existing tests green

### Subagents
- **Implementation:** `senior-fullstack-dev`

---

## Phase v0.8 — Hardware Port

**Goal:** Compile and run on real STM32 and ESP32 hardware with a verified hardware-in-the-loop smoke test.

### Tasks

| # | File | Change |
|---|------|--------|
| 1 | `cmake/toolchains/arm-none-eabi.cmake` | Cross-compile toolchain: `arm-none-eabi-g++`, `-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb`, no OS |
| 2 | `cmake/toolchains/xtensa-esp32.cmake` | ESP-IDF toolchain integration |
| 3 | `include/micromind/platform.h` | New file: `MICROMIND_ASSERT(expr)` macro — `assert()` on host, infinite loop + debug output on bare-metal; `micromind_trap()` for unrecoverable errors |
| 4 | `cmake/ld/stm32f4.ld` | Example linker script for STM32F4: place `weights.h` data in flash (`.rodata`), library code in `.text` |
| 5 | `include/micromind/weight_loader.h` | New file: `WeightLoader` — reads weights from a `const` array at a fixed linker-script address; no filesystem dependency |
| 6 | `tools/hil_test.py` | Hardware-in-the-loop test harness: sends sensor frames over UART, asserts callback fires within configurable timeout |
| 7 | `docs/hardware_port.md` | Porting guide: toolchain setup, linker script customisation, tested boards |

### Success Criteria
- Firmware compiles with zero warnings for both STM32 and ESP32 targets
- `arm-none-eabi-size` output: `.text` + `.rodata` + `.bss` ≤ 32 KB on STM32F4 baseline
- HiL smoke test passes on at least one physical board

### Subagents
- **Architecture & toolchain:** `software-architect`
- **Implementation:** `senior-fullstack-dev`

---

## Phase v1.0 — Production Hardening

**Goal:** Library is safe, documented, and independently verifiable for safety-critical deployment.

### Tasks

| # | File | Change |
|---|------|--------|
| 1 | `cmake/cppcheck.cmake` | Integrate `cppcheck` as a CMake custom target; zero findings at `--error-exitcode=1` level |
| 2 | `docs/misra_deviations.md` | Document which MISRA C++:2008 rules apply; justify any deviations (e.g. templates, `assert`) |
| 3 | `cmake/coverage.cmake` | `lcov`/`gcov` coverage target; CI gate at ≥ 90 % line coverage on library sources |
| 4 | `Doxyfile` | Doxygen config: all public API (`include/micromind/`) documented; generate HTML + XML; zero undocumented warnings |
| 5 | `.github/workflows/ci.yml` | CI pipeline: configure → build (host) → test → static analysis → coverage; second job: cross-compile for ARM (build only); triggered on every PR and push to `main` |
| 6 | `CHANGELOG.md` | Version history from v0.1 |
| 7 | `CONTRIBUTING.md` | Contribution guide: build instructions, code style, test requirements, how to add a new sensor channel |
| 8 | Integration test | Run on real 3D printer sensor data; document results in `docs/validation_report.md` |
| 9 | Release | Tag `v1.0.0`; attach binary size report from `arm-none-eabi-size` to GitHub release |

### Success Criteria
- CI green on every PR
- Binary ≤ 32 KB flash on STM32F4
- Doxygen builds with zero warnings
- `cppcheck` clean
- 90 % line coverage
- `v1.0.0` tag published

### Subagents
- **CI & architecture:** `software-architect`
- **Implementation:** `senior-fullstack-dev`
- **Final audit:** `code-quality-auditor`

---

## Dependency Graph

```
v0.1 (done)
  └── v0.2 (hardening)
        └── v0.3 (multi-layer inference)
              └── v0.4 (multi-sensor input)
                    ├── v0.5 (weight loading)     ← also needs v0.6 output
                    └── v0.6 (training pipeline)  ← feeds v0.5
                          └── v0.7 (hysteresis)
                                └── v0.8 (hardware port)
                                      └── v1.0 (production hardening)
```

v0.5 and v0.6 are developed in parallel; v0.5 C++ side can use placeholder weights while v0.6 trains the real model.

---

## Quick Reference — Subagent Assignment

| Phase | Primary Subagent | Secondary Subagent |
|-------|-----------------|-------------------|
| v0.2 | `senior-fullstack-dev` | `pr-review-toolkit:code-reviewer` |
| v0.3 | `senior-fullstack-dev` | — |
| v0.4 | `senior-fullstack-dev` | — |
| v0.5 | `senior-fullstack-dev` | `senior-python-dev` |
| v0.6 | `senior-python-dev` | — |
| v0.7 | `senior-fullstack-dev` | — |
| v0.8 | `senior-fullstack-dev` | `software-architect` |
| v1.0 | `senior-fullstack-dev` | `software-architect` + `code-quality-auditor` |
