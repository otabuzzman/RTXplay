//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "camera.h"
#include "optics.h"

#include "optixTriangle.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;

extern "C" { __constant__ LpGeneral lp_general ; }

static __forceinline__ __device__ uchar4 sRGB( const float r, const float g, const float b ) {
	return make_uchar4(
		(unsigned char) ( util::clamp( r, .0f, 1.f )*255.f+.5f ),
		(unsigned char) ( util::clamp( g, .0f, 1.f )*255.f+.5f ),
		(unsigned char) ( util::clamp( b, .0f, 1.f )*255.f+.5f ), 255u ) ;
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

	// transform x/y pixel coords (range 0/0 to w/h)
	// into s/t viewport coords (range -1/-1 to 1/1)
	const float s = 2.f*static_cast<float>( idx.x )/static_cast<float>( dim.x )-1.f ;
	const float t = 2.f*static_cast<float>( idx.y )/static_cast<float>( dim.y )-1.f ;

	// get Camera class instance from SBT
	const Camera* camera  = reinterpret_cast<Camera*>( optixGetSbtDataPointer() ) ;

	float3 ori, dir ;
	camera->ray( s, t, ori, dir, &state ) ;

	// payloads to carry back color from abyss of trace
	unsigned int r, g, b ;

	// payloads to propagate RNG state down the trace
	// pointer split due to 32 bit limit of payload values
	unsigned int sh = reinterpret_cast<unsigned long long>( &state )>>32 ;
	unsigned int sl = reinterpret_cast<unsigned long long>( &state )&0x00000000ffffffff ;

	// payload to propagate depth count down the trace
	unsigned int depth = 1 ;

	// shoot initial ray
	optixTrace(
			lp_general.as_handle,
			ori,                        // ray position
			dir,                        // ray direction
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

	// update pixel in image buffer with this ray's color
	lp_general.image[pix] = sRGB(
		__uint_as_float( r ),
		__uint_as_float( g ),
		__uint_as_float( b ) ) ;
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

extern "C" __global__ void __closesthit__diffuse() {
	// retrieve actual trace depth from payload
	unsigned int depth = optixGetPayload_5() ;

	// go deeper as long as not reaching ground
	if ( lp_general.depth>depth ) {
		// retrieve data (SBT record) of thing being hit
		const Optics optics = *reinterpret_cast<Optics*>( optixGetSbtDataPointer() ) ;

		// retrieve index of triangle (primitive) being hit
		// index is same as in OptixBuildInput array handed to optixAccelBuild()
		const int   prix = optixGetPrimitiveIndex() ;
		const uint3 trix = optics.ices[prix] ;

		// use index to access triangle vertices
		const float3 A = optics.vces[trix.x] ;
		const float3 B = optics.vces[trix.y] ;
		const float3 C = optics.vces[trix.z] ;

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
		unsigned int sh = optixGetPayload_3() ; // used as well to propagate PNG further down
		unsigned int sl = optixGetPayload_4() ;
		curandState* state = reinterpret_cast<curandState*>( static_cast<unsigned long long>( sh )<<32|sl ) ;

		// finally the diffuse reflection according to RTOW
		const float3 dir = N+V::rndVon1sphere( state ) ;

		// payloads to carry back color
		unsigned int r, g, b ;

		// one step beyond (recursion)
		optixTrace(
				lp_general.as_handle,
				hit,                        // ray position
				dir,                        // ray direction
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
				++depth  // payload downstream propagation: recursion depth
				) ;

		// update this ray's diffuse color according to RTOW and propagate
		const float3 albedo = optics.diffuse.albedo ;
		optixSetPayload_0( __float_as_uint( albedo.x*__uint_as_float( r ) ) ) ;
		optixSetPayload_1( __float_as_uint( albedo.y*__uint_as_float( g ) ) ) ;
		optixSetPayload_2( __float_as_uint( albedo.z*__uint_as_float( b ) ) ) ;
	} else {
		optixSetPayload_0( 0u ) ;
		optixSetPayload_1( 0u ) ;
		optixSetPayload_2( 0u ) ;
	}
}

extern "C" __global__ void __closesthit__reflect() {
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float3 color = { barycentrics.x, barycentrics.y, .5f };

	optixSetPayload_0( __float_as_uint( color.x ) ) ;
	optixSetPayload_1( __float_as_uint( color.y ) ) ;
	optixSetPayload_2( __float_as_uint( color.z ) ) ;
}

extern "C" __global__ void __closesthit__refract() {
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float3 color = { barycentrics.x, barycentrics.y, 0.f };

	optixSetPayload_0( __float_as_uint( color.x ) ) ;
	optixSetPayload_1( __float_as_uint( color.y ) ) ;
	optixSetPayload_2( __float_as_uint( color.z ) ) ;
}
