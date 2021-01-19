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

#include "v.h"
#include "util.h"

#include "camera.h"

#include "optixTriangle.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;

__forceinline__ __device__ uchar4 sRGB( const float3& c )
{
    return make_uchar4(
        // README.md#Findings#3
        (unsigned char) ( util::clamp( c.x, .0f, 1.f )*255.f+.5f ),
        (unsigned char) ( util::clamp( c.y, .0f, 1.f )*255.f+.5f ),
        (unsigned char) ( util::clamp( c.z, .0f, 1.f )*255.f+.5f ), 255u ) ;
}

// README.md#Findings#2
extern "C" { __constant__ LpGeneral lpGeneral ; }

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // transform x/y pixel ccord (range 0/0 to w/h)
    // into s/t viewport coords (range -1/-1 to 1/1)
    float s = 2.f*static_cast<float>( idx.x )/static_cast<float>( dim.x )-1.f ;
    float t = 2.f*static_cast<float>( idx.y )/static_cast<float>( dim.y )-1.f ;

    { // Camera::ray() code replacement
        float3 ori = lpGeneral.camera.eye ;
        float3 dir = lpGeneral.camera.dist*( s*lpGeneral.camera.wvec+t*lpGeneral.camera.hvec )-lpGeneral.camera.eye ;
    }

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
            lpGeneral.handle,
            ori,
            dir,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2 );
    float3 result;
    result.x = int_as_float( p0 );
    result.y = int_as_float( p1 );
    result.z = int_as_float( p2 );

    // Record results in our output raster
    lpGeneral.image[idx.y * lpGeneral.image_width + idx.x] = sRGB( result );
}

extern "C" __global__ void __miss__ms()
{
    MissData* miss_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    setPayload(  miss_data->bg_color );
}

extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();

    setPayload( make_float3( barycentrics.x, barycentrics.y, 1.0f ) );
}
