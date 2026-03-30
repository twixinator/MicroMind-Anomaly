# MicroMind-Anomaly

Tiny anomaly-detection inference library for embedded 3D printer safety.

## Language & Standard

- **C++14** (overrides global `~/.claude/CLAUDE.md` C++20 convention)
- Compiler: GCC 12+ / Clang 15+ / MSVC 17.5+
- Build: CMake 3.25+, Ninja recommended

## Hard Constraints

| Constraint | Rationale |
|------------|-----------|
| **No heap allocation** | `std::vector`, `new`, `malloc` prohibited. Static arrays and `std::array` only. |
| **No exceptions** | `-fno-exceptions` enforced. No `throw`, no `try`/`catch` in library code. |
| **Single-threaded** | No mutexes, atomics, or thread-local storage. |
| **Weights in ROM** | `static const` arrays placed in `.rodata` at link time. No runtime loading (until v0.5). |
| **Binary size** | Target: <= 32 KB flash (STM32F4 baseline). |

## Build Commands

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/tests/micromind_tests        # or micromind_tests.exe on Windows
```

## Project Structure

```
include/micromind/     # Public headers (config, detector, inference_engine, ring_buffer)
src/                   # Implementation files
tests/                 # GoogleTest unit tests
docs/                  # Design specs (gitignored)
tools/                 # Scripts and utilities (future)
```

## Naming Override

Free functions in the `micromind` namespace use **snake_case** (e.g., `mat_vec_mul`, `relu`, `dense_forward`). This overrides the global PascalCase convention for functions — it matches the existing codebase and is consistent with C++ standard library style, which is more natural for a low-level embedded library.

## Testing

- Framework: GoogleTest 1.14.0 (fetched via CMake FetchContent)
- Tests compile without `-fno-exceptions` (GoogleTest requires exceptions)
- All tests must pass under `-Wall -Wextra -Werror`
