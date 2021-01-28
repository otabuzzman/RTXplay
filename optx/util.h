#ifndef UTIL_H
#define UTIL_H

#include <limits>

namespace util {

const float kInfinty = std::numeric_limits<float>::infinity() ;
const float kPi      = 3.14159265358979323846f ;

__forceinline__ __host__ __device__ float rnd()                                   { return static_cast<float>( rand() )/( static_cast<float>( RAND_MAX )+1.f ) ; }
__forceinline__ __host__ __device__ float rnd( const float min, const float max ) { return min+util::rnd()*( max-min ) ; }

}

#ifdef __CUDACC__
#include "util_device.h"
#else
#include "util_host.h"
#endif

#endif // UTIL_H
