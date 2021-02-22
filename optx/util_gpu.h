#ifndef UTIL_GPU_H
#define UTIL_GPU_H

#include <curand.h>
#include <curand_kernel.h>

#include "frand48.h"

namespace util {

__forceinline__ __device__ float rnd( Frand48* state )                                   { return ( *state )() ; }
__forceinline__ __device__ float rnd( const float min, const float max, Frand48* state ) { return min+rnd( state )*( max-min ) ; }

__forceinline__ __device__ float  clamp( const float   x, const float min, const float max ) { return min>x ? min : x>max ? max : x ; }
__forceinline__ __device__ float3 clamp( const float3& x, const float min, const float max ) { return make_float3( clamp( x.x, min, max ), clamp( x.y, min, max ), clamp( x.z, min, max ) ) ; }

}

#endif // UTIL_GPU_H
