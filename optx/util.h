#ifndef UTIL_H
#define UTIL_H

#include <limits>

#include <vector_functions.h>
#include <vector_types.h>

namespace util {

const float kInfinty = std::numeric_limits<float>::infinity() ;
const float kPi      = 3.14159265358979323846f ;

__forceinline__ __device__ float  clamp( const float   x, const float min, const float max ) { return min>x ? min : x>max ? max : x ; }
__forceinline__ __device__ float3 clamp( const float3& x, const float min, const float max ) { return make_float3( clamp( x.x, min, max ), clamp( x.y, min, max ), clamp( x.z, min, max ) ) ; }

}

#endif // UTIL_H
