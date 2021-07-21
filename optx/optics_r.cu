#include <optix.h>

#include "camera.h"
#include "thing.h"

#include "rtwo.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;

extern "C" { __constant__ LpGeneral lp_general ; }

extern "C" __global__ void __closesthit__diffuse() {
	// retrieve actual trace depth from payload
	unsigned int depth = optixGetPayload_5() ;

	// go deeper as long as not reaching ground
	if ( lp_general.depth>depth ) {
		// retrieve data (SBT record) of thing being hit
		const Thing* thing  = reinterpret_cast<Thing*>( optixGetSbtDataPointer() ) ;
		const Optics optics = thing->optics() ;

		// retrieve index of triangle (primitive) being hit
		// index is same as in OptixBuildInput array handed to optixAccelBuild()
		const int   prix = optixGetPrimitiveIndex() ;
		const uint3 trix = thing->d_ices()[prix] ;

		// use index to access triangle vertices
		const float3* d_vces = thing->d_vces() ;
		const float3 A = d_vces[trix.x] ;
		const float3 B = d_vces[trix.y] ;
		const float3 C = d_vces[trix.z] ;

		// retrieve triangle barycentric coordinates of hit
		const float2 bary = optixGetTriangleBarycentrics() ;
		const float u = bary.x ;
		const float v = bary.y ;
		const float w = 1.f-u-v ;

		// calculate world coordinates of hit
		const float3 hit = w*A+u*B+v*C ;

		// calculate primitive normal
		const float3 d = optixGetWorldRayDirection() ;
		float3 N = V::unitV( V::cross( B-A, C-A ) ) ;
		if ( V::dot( d, N )>0.f )
			N = -N ;

		// assemble RNG pointer from payload
		unsigned int sh = optixGetPayload_3() ; // used as well to propagate RNG further down
		unsigned int sl = optixGetPayload_4() ;
		curandState* state = reinterpret_cast<curandState*>( util::fit64( sh, sl ) ) ;

		// finally the diffuse reflection according to RTOW
		// see CPU version of RTOW, optics.h: Diffuse.spray()
			const float3 dir = N+V::rndVon1sphere( state ) ;
		//

		// payloads to carry back color
		unsigned int r, g, b ;

		// payloads to carry back rays per trace
		unsigned int rpt ;

		// one step beyond (recursion)
		optixTrace(
				lp_general.as_handle,
				hit,                        // next ray's origin
				dir,                        // next ray's direction
				1e-3f,                      // tmin
				1e16f,                      // tmax
				0.f,                        // motion
				OptixVisibilityMask( 255 ),
				OPTIX_RAY_FLAG_NONE,
				0,                          // SBT related
				1,                          // SBT related
				0,                          // SBT related
				r, g, b, // payload upstream propagation: color
				sh, sl,  // payload downstream propagation: RNG state pointer
				++depth, // payload downstream propagation: recursion depth
				rpt      // payload upstream propagation: rays per trace
				) ;

		// update this ray's diffuse color according to RTOW and propagate
		const float3 albedo = optics.diffuse.albedo ;
		optixSetPayload_0( __float_as_uint( albedo.x*__uint_as_float( r ) ) ) ;
		optixSetPayload_1( __float_as_uint( albedo.y*__uint_as_float( g ) ) ) ;
		optixSetPayload_2( __float_as_uint( albedo.z*__uint_as_float( b ) ) ) ;

		optixSetPayload_6( rpt ) ;
	} else {
		optixSetPayload_0( 0u ) ;
		optixSetPayload_1( 0u ) ;
		optixSetPayload_2( 0u ) ;

		optixSetPayload_6( depth+1u ) ;
	}
}

extern "C" __global__ void __closesthit__reflect() {
	// retrieve actual trace depth from payload
	unsigned int depth = optixGetPayload_5() ;

	// retrieve data (SBT record) of thing being hit
	const Thing* thing  = reinterpret_cast<Thing*>( optixGetSbtDataPointer() ) ;
	const Optics optics = thing->optics() ;

	// retrieve index of triangle (primitive) being hit
	// index is same as in OptixBuildInput array handed to optixAccelBuild()
	const int   prix = optixGetPrimitiveIndex() ;
	const uint3 trix = thing->d_ices()[prix] ;

	// use index to access triangle vertices
	const float3* d_vces = thing->d_vces() ;
	const float3 A = d_vces[trix.x] ;
	const float3 B = d_vces[trix.y] ;
	const float3 C = d_vces[trix.z] ;

	// retrieve triangle barycentric coordinates of hit
	const float2 bary = optixGetTriangleBarycentrics() ;
	const float u = bary.x ;
	const float v = bary.y ;
	const float w = 1.f-u-v ;

	// calculate world coordinates of hit
	const float3 hit = w*A+u*B+v*C ;

	// calculate primitive normal
	float3 d = optixGetWorldRayDirection() ;
	float3 N = V::unitV( V::cross( B-A, C-A ) ) ;
	if ( V::dot( d, N )>0.f )
		N = -N ;

	// assemble RNG pointer from payload
	unsigned int sh = optixGetPayload_3() ; // used as well to propagate RNG further down
	unsigned int sl = optixGetPayload_4() ;
	curandState* state = reinterpret_cast<curandState*>( util::fit64( sh, sl ) ) ;

	// finally the reflection according to RTOW
	// see CPU version of RTOW, optics.h:  Reflect.spray()
		const float3 d1V     = V::unitV( d ) ;              // v.h: reflect()
		const float3 reflect = d1V-2.f*V::dot( d1V, N )*N ; // v.h: reflect()
		const float fuzz = optics.reflect.fuzz ;
		const float3 dir = reflect+fuzz*V::rndVin1sphere( state ) ;
	//

	// go deeper as long as not reaching ground and same directions of hit point normal and reflected ray.
	if ( lp_general.depth>depth && V::dot( dir, N )>0 ) {
		// payloads to carry back color
		unsigned int r, g, b ;

		// payloads to carry back rays per trace
		unsigned int rpt ;

		// one step beyond (recursion)
		optixTrace(
				lp_general.as_handle,
				hit,                        // next ray's origin
				dir,                        // next ray's direction
				1e-3f,                      // tmin
				1e16f,                      // tmax
				0.f,                        // motion
				OptixVisibilityMask( 255 ),
				OPTIX_RAY_FLAG_NONE,
				0,                          // SBT related
				1,                          // SBT related
				0,                          // SBT related
				r, g, b, // payload upstream propagation: color
				sh, sl,  // payload downstream propagation: RNG state pointer
				++depth, // payload downstream propagation: recursion depth
				rpt      // payload upstream propagation: rays per trace
				) ;

		// update this ray's reflect color according to RTOW and propagate
		const float3 albedo = optics.reflect.albedo ;
		optixSetPayload_0( __float_as_uint( albedo.x*__uint_as_float( r ) ) ) ;
		optixSetPayload_1( __float_as_uint( albedo.y*__uint_as_float( g ) ) ) ;
		optixSetPayload_2( __float_as_uint( albedo.z*__uint_as_float( b ) ) ) ;

		optixSetPayload_6( rpt ) ;
	} else {
		optixSetPayload_0( 0u ) ;
		optixSetPayload_1( 0u ) ;
		optixSetPayload_2( 0u ) ;

		optixSetPayload_6( depth+1u ) ;
	}
}

extern "C" __global__ void __closesthit__refract() {
	// retrieve actual trace depth from payload
	unsigned int depth = optixGetPayload_5() ;

	// go deeper as long as not reaching ground
	if ( lp_general.depth>depth ) {
		// retrieve data (SBT record) of thing being hit
		const Thing* thing  = reinterpret_cast<Thing*>( optixGetSbtDataPointer() ) ;
		const Optics optics = thing->optics() ;

		// retrieve index of triangle (primitive) being hit
		// index is same as in OptixBuildInput array handed to optixAccelBuild()
		const int   prix = optixGetPrimitiveIndex() ;
		const uint3 trix = thing->d_ices()[prix] ;

		// use index to access triangle vertices
		const float3* d_vces = thing->d_vces() ;
		const float3 A = d_vces[trix.x] ;
		const float3 B = d_vces[trix.y] ;
		const float3 C = d_vces[trix.z] ;

		// retrieve triangle barycentric coordinates of hit
		const float2 bary = optixGetTriangleBarycentrics() ;
		const float u = bary.x ;
		const float v = bary.y ;
		const float w = 1.f-u-v ;

		// calculate world coordinates of hit
		const float3 hit = w*A+u*B+v*C ;

		// calculate primitive normal
		const float3 d = optixGetWorldRayDirection() ;
		float3 N = V::unitV( V::cross( B-A, C-A ) ) ;
		if ( V::dot( d, N )>0.f )
			N = -N ;

		// assemble RNG pointer from payload
		unsigned int sh = optixGetPayload_3() ; // used as well to propagate RNG further down
		unsigned int sl = optixGetPayload_4() ;
		curandState* state = reinterpret_cast<curandState*>( util::fit64( sh, sl ) ) ;

		// finally the refraction according to RTOW
		// see CPU version of RTOW, optics.h: Refract.spray()
			const float3 d1V = V::unitV( d ) ;
			const float cos_theta = fminf( V::dot( -d1V, N ), 1.f ) ;
			const float sin_theta = sqrtf( 1.f-cos_theta*cos_theta ) ;

			const float3 O = hit-thing->center() ;
			const float ratio = 0.f>V::dot( d, O )
					? 1.f/optics.refract.index
					: optics.refract.index ;
			const bool cannot = ratio*sin_theta>1.f ;

			float r0 = ( 1.f-ratio )/( 1.f+ratio ) ; r0 = r0*r0 ;                // optics.h: Refract.schlick()
			const float schlick = r0+( 1.f-r0 )*powf( ( 1.f-cos_theta ), 5.f ) ; // optics.h: Refract.schlick()

			float3 dir ;
			if ( cannot || schlick>util::rnd( state ) ) {
				dir = d1V-2.f*V::dot( d1V, N )*N ;                                        // v.h: reflect()
			} else {
				const float theta   = fminf( V::dot( -d1V, N ), 1.f ) ;                   // v.h: refract()
				const float3 perpen = ratio*( d1V+theta*N ) ;                             // v.h: refract()
				const float3 parall = -sqrtf( fabsf( 1.f-V::dot( perpen, perpen ) ) )*N ; // v.h: refract()
				dir = perpen+parall ;                                                     // v.h: refract()
			}
		//

		// payloads to carry back color
		unsigned int r, g, b ;

		// payloads to carry back rays per trace
		unsigned int rpt ;

		// one step beyond (recursion)
		optixTrace(
				lp_general.as_handle,
				hit,                        // next ray's origin
				dir,                        // next ray's direction
				1e-3f,                      // tmin
				1e16f,                      // tmax
				0.f,                        // motion
				OptixVisibilityMask( 255 ),
				OPTIX_RAY_FLAG_NONE,
				0,                          // SBT related
				1,                          // SBT related
				0,                          // SBT related
				r, g, b, // payload upstream propagation: color
				sh, sl,  // payload downstream propagation: RNG state pointer
				++depth, // payload downstream propagation: recursion depth
				rpt      // payload upstream propagation: rays per trace
				) ;

		// update this ray's refract color according to RTOW and propagate
		const float3 albedo = { 1.f, 1.f, 1.f } ;
		optixSetPayload_0( __float_as_uint( albedo.x*__uint_as_float( r ) ) ) ;
		optixSetPayload_1( __float_as_uint( albedo.y*__uint_as_float( g ) ) ) ;
		optixSetPayload_2( __float_as_uint( albedo.z*__uint_as_float( b ) ) ) ;

		optixSetPayload_6( rpt ) ;
	} else {
		optixSetPayload_0( 0u ) ;
		optixSetPayload_1( 0u ) ;
		optixSetPayload_2( 0u ) ;

		optixSetPayload_6( depth+1u ) ;
	}
}
