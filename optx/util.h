#ifndef UTIL_H
#define UTIL_H

#include <limits>

namespace util {

const float kInfinty = std::numeric_limits<float>::infinity() ;
const float kPi      = 3.14159265358979323846f ;

}

#ifdef __CUDACC__
#include "util_gpu.h"
#else
#include "util_cpu.h"
#endif

#endif // UTIL_H

