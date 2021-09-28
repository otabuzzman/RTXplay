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
	unsigned int depth = optixGetPayload_5()+1u ;

	// return if recursion limit reached
	if ( lp_general.depth == depth ) {
		optixSetPayload_0( 0u ) ;
		optixSetPayload_1( 0u ) ;
		optixSetPayload_2( 0u ) ;

		optixSetPayload_5( depth ) ;

		return ;
	}

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
#ifdef CURAND
	curandState* state = reinterpret_cast<curandState*>( util::fit64( sh, sl ) ) ;
#else
	Frand48* state = reinterpret_cast<Frand48*>( util::fit64( sh, sl ) ) ;
#endif // CURAND

	// assemble DGV pointer from payload
	unsigned int dh = optixGetPayload_6() ;
	unsigned int dl = optixGetPayload_7() ;
	DenoiserGuideValues* dgv = reinterpret_cast<DenoiserGuideValues*>( util::fit64( dh, dl ) ) ;

	// finally the diffuse reflection according to RTOW
	// see CPU version of RTOW, optics.h: Diffuse.spray()
		const float3 dir = N+V::rndVon1sphere( state ) ;
	//

	// payloads to carry back color
	unsigned int r, g, b ;

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
			// payloads
			r, g, b, // up: color
			sh, sl,  // down: RNG state pointer
			depth,   // down: current recursion depth, up: final recursion depth (rays per trace)
			dh, dl   // up: denoiser guide values (DGV)
			) ;

	// update this ray's diffuse color according to RTOW and propagate
	const float3 albedo = optics.diffuse.albedo ;
	optixSetPayload_0( __float_as_uint( albedo.x*__uint_as_float( r ) ) ) ;
	optixSetPayload_1( __float_as_uint( albedo.y*__uint_as_float( g ) ) ) ;
	optixSetPayload_2( __float_as_uint( albedo.z*__uint_as_float( b ) ) ) ;

	optixSetPayload_5( depth ) ;

	// update denoiser guide values on first hit on diffuse
	if ( util::isnull( dgv->normal ) )
		dgv->normal = N ;
	if ( util::isnull( dgv->albedo ) )
		dgv->albedo = albedo ;
}

extern "C" __global__ void __closesthit__reflect() {
	// retrieve actual trace depth from payload
	unsigned int depth = optixGetPayload_5()+1u ;

	// return if recursion limit reached
	if ( lp_general.depth == depth ) {
		optixSetPayload_0( 0u ) ;
		optixSetPayload_1( 0u ) ;
		optixSetPayload_2( 0u ) ;

		optixSetPayload_5( depth ) ;

		return ;
	}

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
#ifdef CURAND
	curandState* state = reinterpret_cast<curandState*>( util::fit64( sh, sl ) ) ;
#else
	Frand48* state = reinterpret_cast<Frand48*>( util::fit64( sh, sl ) ) ;
#endif // CURAND

	// assemble DGV pointer from payload
	unsigned int dh = optixGetPayload_6() ;
	unsigned int dl = optixGetPayload_7() ;
	DenoiserGuideValues* dgv = reinterpret_cast<DenoiserGuideValues*>( util::fit64( dh, dl ) ) ;

	// finally the reflection according to RTOW
	// see CPU version of RTOW, optics.h:  Reflect.spray()
		const float3 d1V     = V::unitV( d ) ;              // v.h: reflect()
		const float3 reflect = d1V-2.f*V::dot( d1V, N )*N ; // v.h: reflect()
		const float fuzz = optics.reflect.fuzz ;
		const float3 dir = reflect+fuzz*V::rndVin1sphere( state ) ;
	//

	// check for different directions of hit point normal and reflected ray
	if ( V::dot( dir, N )>0 ) {
		// payloads to carry back color
		unsigned int r, g, b ;

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
				// payloads
				r, g, b, // up: color
				sh, sl,  // down: RNG state pointer
				depth,   // down: current recursion depth, up: final recursion depth (rays per trace)
				dh, dl   // up: denoiser guide values (DGV)
				) ;

		// update this ray's reflect color according to RTOW and propagate
		const float3 albedo = optics.reflect.albedo ;
		optixSetPayload_0( __float_as_uint( albedo.x*__uint_as_float( r ) ) ) ;
		optixSetPayload_1( __float_as_uint( albedo.y*__uint_as_float( g ) ) ) ;
		optixSetPayload_2( __float_as_uint( albedo.z*__uint_as_float( b ) ) ) ;
	} else {
		optixSetPayload_0( 0u ) ;
		optixSetPayload_1( 0u ) ;
		optixSetPayload_2( 0u ) ;
	}

	optixSetPayload_5( depth ) ;

	// update denoiser guide values on first hit on reflect
	if ( util::isnull( dgv->normal ) )
		dgv->normal = N ;
	if ( util::isnull( dgv->albedo ) )
		dgv->albedo = optics.reflect.albedo ;
}

extern "C" __global__ void __closesthit__refract() {
	// retrieve actual trace depth from payload
	unsigned int depth = optixGetPayload_5()+1u ;

	// return if recursion limit reached
	if ( lp_general.depth == depth ) {
		optixSetPayload_0( 0u ) ;
		optixSetPayload_1( 0u ) ;
		optixSetPayload_2( 0u ) ;

		optixSetPayload_5( depth ) ;

		return ;
	}

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
#ifdef CURAND
	curandState* state = reinterpret_cast<curandState*>( util::fit64( sh, sl ) ) ;
#else
	Frand48* state = reinterpret_cast<Frand48*>( util::fit64( sh, sl ) ) ;
#endif // CURAND

	// assemble DGV pointer from payload
	unsigned int dh = optixGetPayload_6() ;
	unsigned int dl = optixGetPayload_7() ;
	DenoiserGuideValues* dgv = reinterpret_cast<DenoiserGuideValues*>( util::fit64( dh, dl ) ) ;

	// finally the refraction according to RTOW
	// see CPU version of RTOW, optics.h: Refract.spray()
		const float3 d1V = V::unitV( d ) ;
		const float cos_theta = fminf( V::dot( -d1V, N ), 1.f ) ;
		const float sin_theta = sqrtf( 1.f-cos_theta*cos_theta ) ;

		const float3 center = {
			thing->transform()[0*4+3] /* x */,
			thing->transform()[1*4+3] /* y */,
			thing->transform()[2*4+3] /* z */
		} ;
		const float3 O = hit-center ;
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
			// payloads
			r, g, b, // up: color
			sh, sl,  // down: RNG state pointer
			depth,   // down: current recursion depth, up: final recursion depth (rays per trace)
			dh, dl   // up: denoiser guide values (DGV)
			) ;

	// update this ray's refract color according to RTOW and propagate
	const float3 albedo = { 1.f, 1.f, 1.f } ;
	optixSetPayload_0( __float_as_uint( albedo.x*__uint_as_float( r ) ) ) ;
	optixSetPayload_1( __float_as_uint( albedo.y*__uint_as_float( g ) ) ) ;
	optixSetPayload_2( __float_as_uint( albedo.z*__uint_as_float( b ) ) ) ;

	optixSetPayload_5( depth ) ;
}
