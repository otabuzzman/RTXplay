#include <optix.h>

#include "camera.h"
#include "thing.h"

#include "rtwo.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;
using V::operator/ ;

extern "C" { __constant__ LpGeneral lp_general ; }

static __forceinline__ __device__ uchar4 sRGB( const float3& color ) {
	return make_uchar4( // sRGB approximation with gamma 2
		(unsigned char) ( util::clamp( sqrtf( color.x ), .0f, .999f )*256.f ),
		(unsigned char) ( util::clamp( sqrtf( color.y ), .0f, .999f )*256.f ),
		(unsigned char) ( util::clamp( sqrtf( color.z ), .0f, .999f )*256.f ), 255u ) ;
}

extern "C" __global__ void __raygen__camera() {
	// dim.x/ dim.y correspond to image width/ height
	const uint3 dim = optixGetLaunchDimensions() ;
	// idx.x/ idx.y correspond to this pixel
	const uint3 idx = optixGetLaunchIndex() ;

	// calculate pixel index for image buffer access
	const unsigned int pix = dim.x*idx.y+idx.x ;

	// initialize random number generator
#ifdef CURAND
	curandState state ;
	curand_init( 4711, pix, 0, &state ) ;
#else
	Frand48 state ;
	state.init( pix ) ;
#endif // CURAND

	// payloads to carry back color from abyss of trace
	unsigned int r, g, b ;

	// payloads to propagate RNG state down the trace
	// pointer split due to 32 bit limit of payload values
	unsigned int sh, sl ;
	util::cut64( &state, sh, sl ) ;

	// pixel color accumulator
	float3 color = {} ;
	// rays per pixel accumulator
	unsigned int rpp = 0 ;
	for ( int i = 0 ; lp_general.spp>i ; i++ ) {
		// transform x/y pixel coords (range 0/0 to w/h)
		// into s/t viewport coords (range -1/-1 to 1/1)
		const float s = 2.f*static_cast<float>( idx.x+util::rnd( &state ) )/static_cast<float>( dim.x )-1.f ;
		const float t = 2.f*static_cast<float>( idx.y+util::rnd( &state ) )/static_cast<float>( dim.y )-1.f ;

		float3 ori, dir ;
		lp_general.camera.ray( s, t, ori, dir, &state ) ;

		// payload to propagate depth count down and recursion depth up
		unsigned int depth = 0 ;

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
				// payloads
				r, g, b, // up: color
				sh, sl,  // down: RNG state pointer
				depth    // down: current recursion depth, up: final recursion depth (rays per trace)
				) ;

		// accumulate this ray's color
		color = color+make_float3( __uint_as_float( r ), __uint_as_float( g ), __uint_as_float( b ) ) ;
		// accumulate rays per pixel
		rpp = rpp+depth ;
	}

	// update pixel in image buffer with mean color
	lp_general.image[pix] = sRGB( color/lp_general.spp ) ;
	// save rpp to respective buffer
	lp_general.rpp[pix] = rpp ;
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

	optixSetPayload_5( optixGetPayload_5()+1 ) ;
}
