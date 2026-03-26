enable_testing()

add_executable(micromind_tests
    test_ring_buffer.cpp
    test_inference.cpp
)

# Apply warning flags to test sources (but NOT -fno-exceptions — GoogleTest needs exceptions).
if(NOT MSVC)
    target_compile_options(micromind_tests PRIVATE -Wall -Wextra -Werror)
endif()

target_link_libraries(micromind_tests
    PRIVATE
        micromind
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(micromind_tests)
