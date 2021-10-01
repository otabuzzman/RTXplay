// system includes
// none

// subsystem includes
// OptiX
#include <optix.h>

// local includes
#include "camera.h"
#include "rtwo.h"
#include "thing.h"

// file specific includes
// none

using V::operator+ ;
using V::operator- ;
using V::operator* ;
using V::operator*= ;
using V::operator/ ;

extern "C" { __constant__ LpGeneral lp_general ; }

extern "C" __global__ void __raygen__camera() {
	// dim.x/ dim.y correspond to image width/ height
	const uint3 dim = optixGetLaunchDimensions() ;
	// idx.x/ idx.y correspond to this pixel
	const uint3 idx = optixGetLaunchIndex() ;

	// calculate pixel index for image buffer access
	const unsigned int pix = dim.x*idx.y+idx.x ;

	RayParam rayparam = {} ;
	// payload to propagate ray parameter along the trace
	// pointer split due to 32 bit limit of payload values
	unsigned int ph, pl ;
	util::cut64( &rayparam, ph, pl ) ;

	// initialize random number generator
#ifdef CURAND
	curand_init( 4711, pix, 0, &rayparam.rng ) ;
#else
	rayparam.rng.init( pix ) ;
#endif // CURAND

	// color accumulator
	float3 color = {} ;
	// rays per pixel accumulator
	unsigned int rpp = 0 ;
	// denoiser guide value accumulators
	float3 normal = {} ;
	float3 albedo = {} ;
	for ( int i = 0 ; lp_general.spp>i ; i++ ) {
		// transform x/y pixel coords (range 0/0 to w/h)
		// into s/t viewport coords (range -1/-1 to 1/1)
		const float s = 2.f*static_cast<float>( idx.x+util::rnd( &rayparam.rng ) )/static_cast<float>( dim.x )-1.f ;
		const float t = 2.f*static_cast<float>( idx.y+util::rnd( &rayparam.rng ) )/static_cast<float>( dim.y )-1.f ;

		float3 ori, dir ;
		lp_general.camera.ray( s, t, ori, dir, &rayparam.rng ) ;

		rayparam.color  = { 1.f, 1.f, 1.f } ;
		rayparam.hit    = ori ;
		rayparam.dir    = dir ;
		rayparam.stat   = RP_STAT_CONT ;
		rayparam.normal = {} ;
		rayparam.albedo = {} ;

		unsigned int depth = 0 ;
		while ( lp_general.depth>depth && rayparam.stat == RP_STAT_CONT ) {
			// the ray gun
			optixTrace(
					lp_general.as_handle,
					rayparam.hit,               // next ray's origin
					rayparam.dir,               // next ray's direction
					1e-3f,                      // tmin
					1e16f,                      // tmax
					0.f,                        // motion
					OptixVisibilityMask( 255 ),
					OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
					0,                          // SBT related
					1,                          // SBT related
					0,                          // SBT related
					ph, pl                      // payload: ray parameter
					) ;
			depth++ ;
		}

		// accumulate this trace's colors
		color = color+rayparam.color ;
		// accumulate rays per trace
		rpp = rpp+depth ;

		// acccumulate this trace's normals and albedos
		normal = normal+rayparam.normal ;
		albedo = albedo+rayparam.albedo ;
	}

	// update pixel in image buffer with mean color
	lp_general.rawRGB[pix] = util::clamp( color/lp_general.spp, 0.f, 1.f ) ;
	// save rpp at respective buffer index
	lp_general.rpp[pix] = rpp ;

	// update denoiser guide layers if appropriate
	if ( lp_general.normals )
		lp_general.normals[pix] = normal/lp_general.spp ;
	if ( lp_general.albedos )
		lp_general.albedos[pix] = albedo/lp_general.spp ;
}

extern "C" __global__ void __miss__ambient() {
	// retrieve ray parameter from payload
	unsigned int ph = optixGetPayload_0() ;
	unsigned int pl = optixGetPayload_1() ;
	RayParam* rayparam = reinterpret_cast<RayParam*>( util::fit64( ph, pl ) ) ;

	// get ambient color from MS program group's SBT record
	const float3 ambient = *reinterpret_cast<float3*>( optixGetSbtDataPointer() ) ;

	// get this ray's direction from OptiX and normalize
	const float3 unit    = V::unitV( optixGetWorldRayDirection() ) ;

	// calculate background color according to RTOW
	const float t        = .5f*( unit.y+1.f ) ;
	const float3 white   = { 1.f, 1.f, 1.f } ;

	rayparam->color *= ( 1.f-t )*white+t*ambient ;
	rayparam->stat   = RP_STAT_MISS ;
}
