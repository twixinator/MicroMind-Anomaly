// inference_engine.cpp — intentionally minimal.
//
// All inference_engine functions (mat_vec_mul, relu) are C++14 function
// templates defined inline in inference_engine.h. Template definitions must
// be visible at every instantiation site, so there is nothing to compile
// here. This translation unit exists solely as a CMake anchor: it gives the
// 'micromind' static library target a concrete .cpp file to build, which
// satisfies CMake's requirement that STATIC library targets have at least one
// source file.
//
// If the inference engine ever gains non-template functions (e.g. a runtime
// model loader), their definitions belong here.

#include "micromind/inference_engine.h"
