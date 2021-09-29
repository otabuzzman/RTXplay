#include <optix.h>

#include "camera.h"
#include "thing.h"

#include "rtwo.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;
using V::operator*= ;

extern "C" { __constant__ LpGeneral lp_general ; }

extern "C" __global__ void __closesthit__diffuse() {
	// retrieve ray parameter from payload
	unsigned int ph = optixGetPayload_0() ;
	unsigned int pl = optixGetPayload_1() ;
	RayParam* rayparam = reinterpret_cast<RayParam*>( util::fit64( ph, pl ) ) ;

	// retrieve data (SBT record) of thing being hit
	const Thing* thing  = reinterpret_cast<Thing*>( optixGetSbtDataPointer() ) ;
	const Optics optics = thing->optics() ;

	// retrieve index of triangle (primitive) being hit
	// index is same as in OptixBuildInput array handed to optixAccelBuild()
	const int   prix = optixGetPrimitiveIndex() ;
	const uint3 trix = thing->d_ices()[prix] ;

	// use index to access triangle vertices
	const float3* d_vces = thing->d_vces() ;
	const float3 a = d_vces[trix.x] ;
	const float3 p = d_vces[trix.y] ;
	const float3 c = d_vces[trix.z] ;
	// apply AS preTransform matrix
	const float* m = thing->transform() ;
	const float3 A = {
		a.x*m[0*4+0]+a.y*m[0*4+1]+a.z*m[0*4+2]+m[0*4+3],
		a.x*m[1*4+0]+a.y*m[1*4+1]+a.z*m[1*4+2]+m[1*4+3],
		a.x*m[2*4+0]+a.y*m[2*4+1]+a.z*m[2*4+2]+m[2*4+3]
	} ;
	const float3 B = {
		p.x*m[0*4+0]+p.y*m[0*4+1]+p.z*m[0*4+2]+m[0*4+3],
		p.x*m[1*4+0]+p.y*m[1*4+1]+p.z*m[1*4+2]+m[1*4+3],
		p.x*m[2*4+0]+p.y*m[2*4+1]+p.z*m[2*4+2]+m[2*4+3]
	} ;
	const float3 C = {
		c.x*m[0*4+0]+c.y*m[0*4+1]+c.z*m[0*4+2]+m[0*4+3],
		c.x*m[1*4+0]+c.y*m[1*4+1]+c.z*m[1*4+2]+m[1*4+3],
		c.x*m[2*4+0]+c.y*m[2*4+1]+c.z*m[2*4+2]+m[2*4+3]
	} ;

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

	// finally the diffuse reflection according to RTOW
	// see CPU version of RTOW, optics.h: Diffuse.spray()
		const float3 dir = N+V::rndVon1sphere( &rayparam->rng ) ;
	//

	rayparam->hit = hit ;
	rayparam->dir = dir ;

	// update this ray's diffuse color according to RTOW and propagate
	rayparam->color *= optics.diffuse.albedo ;

	rayparam->stat = RP_STAT_CONT ;

	// update denoiser guide values on first hit on diffuse
	if ( util::isnull( rayparam->normal ) )
		rayparam->normal = N ;
	if ( util::isnull( rayparam->albedo ) )
		rayparam->albedo = optics.diffuse.albedo ;
}

extern "C" __global__ void __closesthit__reflect() {
	// retrieve ray parameter from payload
	unsigned int ph = optixGetPayload_0() ;
	unsigned int pl = optixGetPayload_1() ;
	RayParam* rayparam = reinterpret_cast<RayParam*>( util::fit64( ph, pl ) ) ;

	// retrieve data (SBT record) of thing being hit
	const Thing* thing  = reinterpret_cast<Thing*>( optixGetSbtDataPointer() ) ;
	const Optics optics = thing->optics() ;

	// retrieve index of triangle (primitive) being hit
	// index is same as in OptixBuildInput array handed to optixAccelBuild()
	const int   prix = optixGetPrimitiveIndex() ;
	const uint3 trix = thing->d_ices()[prix] ;

	// use index to access triangle vertices
	const float3* d_vces = thing->d_vces() ;
	const float3 a = d_vces[trix.x] ;
	const float3 p = d_vces[trix.y] ;
	const float3 c = d_vces[trix.z] ;
	// apply AS preTransform matrix
	const float* m = thing->transform() ;
	const float3 A = {
		a.x*m[0*4+0]+a.y*m[0*4+1]+a.z*m[0*4+2]+m[0*4+3],
		a.x*m[1*4+0]+a.y*m[1*4+1]+a.z*m[1*4+2]+m[1*4+3],
		a.x*m[2*4+0]+a.y*m[2*4+1]+a.z*m[2*4+2]+m[2*4+3]
	} ;
	const float3 B = {
		p.x*m[0*4+0]+p.y*m[0*4+1]+p.z*m[0*4+2]+m[0*4+3],
		p.x*m[1*4+0]+p.y*m[1*4+1]+p.z*m[1*4+2]+m[1*4+3],
		p.x*m[2*4+0]+p.y*m[2*4+1]+p.z*m[2*4+2]+m[2*4+3]
	} ;
	const float3 C = {
		c.x*m[0*4+0]+c.y*m[0*4+1]+c.z*m[0*4+2]+m[0*4+3],
		c.x*m[1*4+0]+c.y*m[1*4+1]+c.z*m[1*4+2]+m[1*4+3],
		c.x*m[2*4+0]+c.y*m[2*4+1]+c.z*m[2*4+2]+m[2*4+3]
	} ;

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

	// finally the reflection according to RTOW
	// see CPU version of RTOW, optics.h:  Reflect.spray()
		const float3 d1V     = V::unitV( d ) ;              // v.h: reflect()
		const float3 reflect = d1V-2.f*V::dot( d1V, N )*N ; // v.h: reflect()
		const float fuzz = optics.reflect.fuzz ;
		const float3 dir = reflect+fuzz*V::rndVin1sphere( &rayparam->rng ) ;
	//

	rayparam->hit = hit ;
	rayparam->dir = dir ;

	// update color dependent of hit point normal and reflected ray direction
	if ( V::dot( dir, N )>0 ) {
		rayparam->color *= optics.reflect.albedo ;
		rayparam->stat   = RP_STAT_CONT ;
	} else {
		rayparam->color  = { 0.f, 0.f, 0.f } ;
		rayparam->stat   = RP_STAT_STOP ;
	}

	// update denoiser guide values on first hit on reflect
	if ( util::isnull( rayparam->normal ) )
		rayparam->normal = N ;
	if ( util::isnull( rayparam->albedo ) )
		rayparam->albedo = optics.reflect.albedo ;
}

extern "C" __global__ void __closesthit__refract() {
	// retrieve ray parameter from payload
	unsigned int ph = optixGetPayload_0() ;
	unsigned int pl = optixGetPayload_1() ;
	RayParam* rayparam = reinterpret_cast<RayParam*>( util::fit64( ph, pl ) ) ;

	// retrieve data (SBT record) of thing being hit
	const Thing* thing  = reinterpret_cast<Thing*>( optixGetSbtDataPointer() ) ;
	const Optics optics = thing->optics() ;

	// retrieve index of triangle (primitive) being hit
	// index is same as in OptixBuildInput array handed to optixAccelBuild()
	const int   prix = optixGetPrimitiveIndex() ;
	const uint3 trix = thing->d_ices()[prix] ;

	// use index to access triangle vertices
	const float3* d_vces = thing->d_vces() ;
	const float3 a = d_vces[trix.x] ;
	const float3 p = d_vces[trix.y] ;
	const float3 c = d_vces[trix.z] ;
	// apply AS preTransform matrix
	const float* m = thing->transform() ;
	const float3 A = {
		a.x*m[0*4+0]+a.y*m[0*4+1]+a.z*m[0*4+2]+m[0*4+3],
		a.x*m[1*4+0]+a.y*m[1*4+1]+a.z*m[1*4+2]+m[1*4+3],
		a.x*m[2*4+0]+a.y*m[2*4+1]+a.z*m[2*4+2]+m[2*4+3]
	} ;
	const float3 B = {
		p.x*m[0*4+0]+p.y*m[0*4+1]+p.z*m[0*4+2]+m[0*4+3],
		p.x*m[1*4+0]+p.y*m[1*4+1]+p.z*m[1*4+2]+m[1*4+3],
		p.x*m[2*4+0]+p.y*m[2*4+1]+p.z*m[2*4+2]+m[2*4+3]
	} ;
	const float3 C = {
		c.x*m[0*4+0]+c.y*m[0*4+1]+c.z*m[0*4+2]+m[0*4+3],
		c.x*m[1*4+0]+c.y*m[1*4+1]+c.z*m[1*4+2]+m[1*4+3],
		c.x*m[2*4+0]+c.y*m[2*4+1]+c.z*m[2*4+2]+m[2*4+3]
	} ;

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

	// finally the refraction according to RTOW
	// see CPU version of RTOW, optics.h: Refract.spray()
		const float3 d1V = V::unitV( d ) ;
		const float cos_theta = fminf( V::dot( -d1V, N ), 1.f ) ;
		const float sin_theta = sqrtf( 1.f-cos_theta*cos_theta ) ;

		const float3 center = {
			m[0*4+3] /* x */,
			m[1*4+3] /* y */,
			m[2*4+3] /* z */
		} ;
		const float3 O = hit-center ;
		const float ratio = 0.f>V::dot( d, O )
				? 1.f/optics.refract.index
				: optics.refract.index ;
		const bool cannot = ratio*sin_theta>1.f ;

		float r0 = ( 1.f-ratio )/( 1.f+ratio ) ; r0 = r0*r0 ;                // optics.h: Refract.schlick()
		const float schlick = r0+( 1.f-r0 )*powf( ( 1.f-cos_theta ), 5.f ) ; // optics.h: Refract.schlick()

		float3 dir ;
		if ( cannot || schlick>util::rnd( &rayparam->rng ) ) {
			dir = d1V-2.f*V::dot( d1V, N )*N ;                                        // v.h: reflect()
		} else {
			const float theta   = fminf( V::dot( -d1V, N ), 1.f ) ;                   // v.h: refract()
			const float3 perpen = ratio*( d1V+theta*N ) ;                             // v.h: refract()
			const float3 parall = -sqrtf( fabsf( 1.f-V::dot( perpen, perpen ) ) )*N ; // v.h: refract()
			dir = perpen+parall ;                                                     // v.h: refract()
		}
	//

	rayparam->hit = hit ;
	rayparam->dir = dir ;

	rayparam->stat = RP_STAT_CONT ;
}
