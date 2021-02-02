#ifndef UTIL_GPU_H
#define UTIL_GPU_H

namespace util {

__forceinline__ __device__ float  clamp( const float   x, const float min, const float max ) { return min>x ? min : x>max ? max : x ; }
__forceinline__ __device__ float3 clamp( const float3& x, const float min, const float max ) { return make_float3( clamp( x.x, min, max ), clamp( x.y, min, max ), clamp( x.z, min, max ) ) ; }

__forceinline__ __device__ static void optxSetPayload( const float3* color ) {
	optixSetPayload_0( float_as_int( color->x ) ) ;
	optixSetPayload_1( float_as_int( color->y ) ) ;
	optixSetPayload_2( float_as_int( color->z ) ) ;
}

}

#endif // UTIL_GPU_H
