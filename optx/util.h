#ifndef UTIL_H
#define UTIL_H

#include <limits>
#include <memory>

#ifndef __CUDACC__
using std::shared_ptr ;
using std::make_shared ;
#endif

namespace util {

const float kInfinty = std::numeric_limits<float>::infinity() ;
const float kPi      = 3.14159265358979323846f ;

__forceinline__ __host__ __device__ float rnd()                                   { return static_cast<float>( rand() )/( static_cast<float>( RAND_MAX )+1.f ) ; }
__forceinline__ __host__ __device__ float rnd( const float min, const float max ) { return min+util::rnd()*( max-min ) ; }

__forceinline__ __device__ float  clamp( const float   x, const float min, const float max ) { return min>x ? min : x>max ? max : x ; }
__forceinline__ __device__ float3 clamp( const float3& x, const float min, const float max ) { return make_float3( clamp( x.x, min, max ), clamp( x.y, min, max ), clamp( x.z, min, max ) ) ; }

#ifdef __CUDACC__
static __forceinline__ __device__ void optxSetPayload( const float3& color ) {
	optixSetPayload_0( float_as_int( color.x ) ) ;
	optixSetPayload_1( float_as_int( color.y ) ) ;
	optixSetPayload_2( float_as_int( color.z ) ) ;
}
#endif

}

#endif // UTIL_H
