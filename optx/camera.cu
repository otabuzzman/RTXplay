#include <optix.h>

#include "camera.h"
#include "thing.h"

#include "rtwo.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;

extern "C" { __constant__ LpGeneral lp_general ; }

static __forceinline__ __device__ uchar4 sRGB( const float3& color, const int spp ) {
	const float s = 1.f/spp ;

	return make_uchar4(
		(unsigned char) ( util::clamp( sqrtf( s*color.x ), .0f, 1.f )*255.f+.5f ),
		(unsigned char) ( util::clamp( sqrtf( s*color.y ), .0f, 1.f )*255.f+.5f ),
		(unsigned char) ( util::clamp( sqrtf( s*color.z ), .0f, 1.f )*255.f+.5f ), 255u ) ;
}

extern "C" __global__ void __raygen__camera() {
	// dim.x/ dim.y correspond to image width/ height
	const uint3 dim = optixGetLaunchDimensions() ;
	// idx.x/ idx.y correspond to this pixel
	const uint3 idx = optixGetLaunchIndex() ;

	// calculate pixel index for image buffer access
	const unsigned int pix = dim.x*idx.y+idx.x ;

	// initialize random number generator
	curandState state ;
	curand_init( 4711, pix, 0, &state ) ;

	// payloads to carry back color from abyss of trace
	unsigned int r, g, b ;

	// payloads to propagate RNG state down the trace
	// pointer split due to 32 bit limit of payload values
	unsigned int sh = reinterpret_cast<unsigned long long>( &state )>>32 ;
	unsigned int sl = reinterpret_cast<unsigned long long>( &state )&0x00000000ffffffff ;

	// payload to propagate depth count down the trace
	unsigned int depth = 0 ;

	// pixel color accumulator
	float3 color = {} ;
	for ( int i = 0 ; lp_general.spp>i ; i++ ) {
		// transform x/y pixel coords (range 0/0 to w/h)
		// into s/t viewport coords (range -1/-1 to 1/1)
		const float s = 2.f*static_cast<float>( idx.x+util::rnd( &state ) )/static_cast<float>( dim.x )-1.f ;
		const float t = 2.f*static_cast<float>( idx.y+util::rnd( &state ) )/static_cast<float>( dim.y )-1.f ;

		// get Camera class instance from SBT
		const Camera* camera  = reinterpret_cast<Camera*>( optixGetSbtDataPointer() ) ;

		float3 ori, dir ;
		camera->ray( s, t, ori, dir, &state ) ;

		// shoot initial ray
		optixTrace(
				lp_general.as_handle,
				ori,                        // next ray's origin
				dir,                        // next ray's direction
				1e-3f,                      // tmin
				1e16f,                      // tmax
				0.f,                        // motion
				OptixVisibilityMask( 255 ),
				OPTIX_RAY_FLAG_NONE,
				0,                          // SBT related
				1,                          // SBT related
				0,                          // SBT related
				r, g, b, // payload upstream: color
				sh, sl,  // payload downstream: RNG state pointer
				depth    // payload downstream: recursion depth
				) ;

		// acccumulate this ray's color
		color = color+make_float3( __uint_as_float( r ), __uint_as_float( g ), __uint_as_float( b ) ) ;
	}

	// update pixel in image buffer with mean color
	lp_general.image[pix] = sRGB( color, lp_general.spp ) ;
}

extern "C" __global__ void __miss__ambient() {
	// get ambient color from MS program group's SBT record
	const float3 ambient = *reinterpret_cast<float3*>( optixGetSbtDataPointer() ) ;

	// get this ray's direction from OptiX and normalize
	const float3 unit    = V::unitV( optixGetWorldRayDirection() ) ;

	// calculate background color according to RTOW
	const float t        = .5f*( unit.y+1.f ) ;
	const float3 white   = { 1.f, 1.f, 1.f } ;
	const float3 color   = ( 1.f-t )*white+t*ambient ;

	optixSetPayload_0( __float_as_uint( color.x ) ) ;
	optixSetPayload_1( __float_as_uint( color.y ) ) ;
	optixSetPayload_2( __float_as_uint( color.z ) ) ;
}
