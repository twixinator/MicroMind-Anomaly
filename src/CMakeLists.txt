add_library(micromind STATIC
    inference_engine.cpp
    detector.cpp
)

target_include_directories(micromind
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(micromind
    PRIVATE micromind_flags
)
