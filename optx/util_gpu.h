#ifndef UTIL_GPU_H
#define UTIL_GPU_H

#include <curand.h>
#include <curand_kernel.h>

namespace util {

__forceinline__ __device__ float rnd( curandState* state )                                   { return static_cast<float>( curand( state ) )/( static_cast<float>( std::numeric_limits<unsigned int>::max() )+1.f ) ; }
__forceinline__ __device__ float rnd( const float min, const float max, curandState* state ) { return min+rnd( state )*( max-min ) ; }

__forceinline__ __device__ float  clamp( const float   x, const float min, const float max ) { return min>x ? min : x>max ? max : x ; }
__forceinline__ __device__ float3 clamp( const float3& x, const float min, const float max ) { return make_float3( clamp( x.x, min, max ), clamp( x.y, min, max ), clamp( x.z, min, max ) ) ; }

__forceinline__ __device__ static void optxSetPayload( const float3* color ) {
	optixSetPayload_0( float_as_int( color->x ) ) ;
	optixSetPayload_1( float_as_int( color->y ) ) ;
	optixSetPayload_2( float_as_int( color->z ) ) ;
}

}

#endif // UTIL_GPU_H
