cmake_minimum_required(VERSION 3.14)
project(MicroMind-Anomaly LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Project-wide compiler flags as an INTERFACE library.
# Scoped to project targets only -- does NOT affect GoogleTest.
# ---------------------------------------------------------------------------
add_library(micromind_flags INTERFACE)
if(NOT MSVC)
    target_compile_options(micromind_flags INTERFACE
        -Wall -Wextra -Werror -fno-exceptions
    )
endif()

# ---------------------------------------------------------------------------
# GoogleTest via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# Prevent GoogleTest from overriding our compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(tests)
