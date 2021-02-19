#ifndef UTIL_H
#define UTIL_H

#include <limits>

#define K_PI 3.14159265358979323846f

namespace util {

const float kInfinity = std::numeric_limits<float>::infinity() ;
const float kNear0    = 1e-8f ;
const float kAcne0    = 1e-3f ;

}

#ifdef __CUDACC__
#include "util_gpu.h" // used when compiling CUDA code (.cu files)
#else
#include "util_cpu.h" // used when compiling host code (.cxx files)
#endif // __CUDACC__

#endif // UTIL_H
