#ifndef UTIL_GPU_H
#define UTIL_GPU_H

#ifdef CURAND
#include <curand_kernel.h>
#else
#include "frand48.h"
#endif // CURAND

namespace util {

#ifdef CURAND
__forceinline__ __device__ float rnd( curandState* state )                                   { return static_cast<float>( curand( state ) )/( static_cast<float>( UINT_MAX )+1.f ) ; }
__forceinline__ __device__ float rnd( const float min, const float max, curandState* state ) { return min+rnd( state )*( max-min ) ; }
#else
__forceinline__ __device__ float rnd( Frand48* state )                                       { return ( *state )() ; }
__forceinline__ __device__ float rnd( const float min, const float max, Frand48* state )     { return min+rnd( state )*( max-min ) ; }
#endif // CURAND

__forceinline__ __device__ float rad( const float deg ) { return deg*kPi/180.f ; }

__forceinline__ __device__ float  clamp( const float   x, const float min, const float max ) { return min>x ? min : x>max ? max : x ; }
__forceinline__ __device__ float3 clamp( const float3& x, const float min, const float max ) { const float3 r{ clamp( x.x, min, max ), clamp( x.y, min, max ), clamp( x.z, min, max ) } ; return r ; }

__forceinline__ __device__ void  cut64( const void* addr64, unsigned int& h32, unsigned int& l32 ) { const unsigned long long int64 = reinterpret_cast<unsigned long long>( addr64 ) ; h32 = int64 >> 32 ; l32 = int64&0x00000000ffffffff ; }
__forceinline__ __device__ void* fit64( const unsigned int h32, const unsigned int l32 )           { return reinterpret_cast<void*>( static_cast<unsigned long long>( h32 ) << 32|l32 ) ; }

__forceinline__ __device__ bool isnull( const float3 x ) { return x.x == 0.f && x.y == 0.f && x.z == 0.f ; }

}

#endif // UTIL_GPU_H
