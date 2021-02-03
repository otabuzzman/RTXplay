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
#include "util.h"
#include "v.h"

#include "optixTriangle.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;

extern "C" { __constant__ LpGeneral lp_general ; }

static __forceinline__ __device__ uchar4 sRGB( const float3& color ) {
	return make_uchar4(
		(unsigned char) ( util::clamp( color.x, .0f, 1.f )*255.f+.5f ),
		(unsigned char) ( util::clamp( color.y, .0f, 1.f )*255.f+.5f ),
		(unsigned char) ( util::clamp( color.z, .0f, 1.f )*255.f+.5f ), 255u ) ;
}

extern "C" __global__ void __raygen__camera() {
	const uint3 idx = optixGetLaunchIndex() ;
	const uint3 dim = optixGetLaunchDimensions() ;

	// transform x/y pixel coords (range 0/0 to w/h)
	// into s/t viewport coords (range -1/-1 to 1/1)
	const float s = 2.f*static_cast<float>( idx.x )/static_cast<float>( dim.x )-1.f ;
	const float t = 2.f*static_cast<float>( idx.y )/static_cast<float>( dim.y )-1.f ;

	// get Camera class instance from SBT
	const Camera* camera  = reinterpret_cast<Camera*>( optixGetSbtDataPointer() ) ;

	float3 ori, dir ;
	camera->ray( s, t, ori, dir ) ;

	// trace into scene
	unsigned int r, g, b ;
	optixTrace(
			lp_general.as_handle,
			ori,
			dir,
			0.f,                        // Min intersection distance
			1e16f,                      // Max intersection distance
			0.f,                        // rayTime -- used for motion blur
			OptixVisibilityMask( 255 ), // Specify always visible
			OPTIX_RAY_FLAG_NONE,
			0,                          // SBT offset   -- See SBT discussion
			1,                          // SBT stride   -- See SBT discussion
			0,                          // missSBTIndex -- See SBT discussion
			r, g, b
			) ;
	const float3 color = make_float3(
			int_as_float( r ),
			int_as_float( g ),
			int_as_float( b )
			) ;

	lp_general.image[lp_general.image_w*idx.y+idx.x] = sRGB( color ) ;
}

extern "C" __global__ void __miss__ambient() {
	// get ambient color from MS program group's SBT record
	const float3 ambient = *reinterpret_cast<float3*>( optixGetSbtDataPointer() ) ;

	// get this ray's direction from OptiX and normalize
	const float3 unit    = V::unitV( optixGetWorldRayDirection() ) ;

	const float t        = .5f*( unit.y+1.f ) ;
	const float3 white   = { 1.f, 1.f, 1.f } ;
	const float3 color   = ( 1.f-t )*white+t*ambient ;

	util::optxSetPayload( &color ) ;
}

extern "C" __global__ void __closesthit__diffuse() {
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float3 color = { barycentrics.x, barycentrics.y, 1.0f };

    util::optxSetPayload( &color );
}

extern "C" __global__ void __closesthit__reflect() {
}

extern "C" __global__ void __closesthit__refract() {
}
