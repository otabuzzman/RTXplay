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

#include <array>
#include <string>
#include <vector>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <sutil/Exception.h>

#include <vector_functions.h>
#include <vector_types.h>

#include "camera.h"
#include "optics.h"
#include "sphere.h"
#include "things.h"
#include "util.h"
#include "v.h"

#include "optixTriangle.h"

using V::operator- ;
using V::operator* ;

const Things scene() {
	Optics o ;
	Things s ;

	o.diffuse.albedo = { .5f, .5f, .5f } ;
	s.push_back( std::make_shared<Sphere>( make_float3( 0.f, -1000.f, 0.f ), 1000.f, o ) ) ;

	for ( int a = -11 ; a<11 ; a++ ) {
		for ( int b = -11 ; b<11; b++ ) {
			auto select = util::rnd() ;
			float3 center = make_float3( a+.9f*util::rnd(), .2f, b+.9f*util::rnd() ) ;
			if ( V::len( center-make_float3( 4.f, .2f, 0.f ) )>.9f ) {
				if ( select<.8f ) {
					o.diffuse.albedo = V::rnd()*V::rnd() ;
					s.push_back( std::make_shared<Sphere>( center, .2f, o ) ) ;
				} else if ( select<.95f ) {
					o.reflect.albedo = V::rnd( .5f, 1.f ) ;
					o.reflect.fuzz = util::rnd( 0.f, .5f ) ;
					s.push_back( std::make_shared<Sphere>( center, .2f, o ) ) ;
				} else {
					o.refract.index = 1.5f ;
					s.push_back( std::make_shared<Sphere>( center, .2f, o ) ) ;
				}
			}
		}
	}

	o.refract.index  = 1.5f ;
	s.push_back( std::make_shared<Sphere>( make_float3(  0.f, 1.f, 0.f ), 1.f, o ) ) ;
	o.diffuse.albedo = { .4f, .2f, .1f } ;
	s.push_back( std::make_shared<Sphere>( make_float3( -4.f, 1.f, 0.f ), 1.f, o ) ) ;
	o.reflect.albedo = { .7f, .6f, .5f } ;
	o.reflect.fuzz   = 0.f ;
	s.push_back( std::make_shared<Sphere>( make_float3(  4.f, 1.f, 0.f ), 1.f, o ) ) ;

	return s ;
}

int main() {
	Things things = scene() ;

	float aspratio = 3.f/2.f ;

	SbtRecordRG sbt_record_camera ;
	sbt_record_camera.data.set(
		{13.f, 2.f, 3.f} /*eye*/,
		{ 0.f, 0.f, 0.f} /*pat*/,
		{ 0.f, 1.f, 0.f} /*vup*/,
		20.f /*fov*/,
		aspratio,
		.1f  /*aperture*/,
		10.f /*distance*/ ) ;

	const int w = 1200 ;                           // image width in pixels
	const int h = static_cast<int>( w/aspratio ) ; // image height in pixels

	SbtRecordMS sbt_record_ambient ;
	sbt_record_ambient.data = { .5f, .7f, 1.f } ;

	try {
		util::optxInitialize() ;

		util::optxBuildAccelerationStructure( things ) ;

		util::optxCreateModules() ;
		util::optxCreateProgramGroups() ;
		util::optxLinkPipeline() ;

		util::optxBuildShaderBindingTable( things ) ;

		std::vector<uchar4> image = util::optxLaunchPipeline( w, h ) ;

		// output image
		{
			std::cout
				<< "P3\n" // magic PPM header
				<< w << ' ' << h << '\n' << 255 << '\n' ;

			for ( int y = h-1 ; y>=0 ; --y ) {
				for ( int x = 0 ; x<w ; ++x ) {
					auto p = image.data()[w*y+x] ;
					std::cout
						<< static_cast<int>( p.x ) << ' '
						<< static_cast<int>( p.y ) << ' '
						<< static_cast<int>( p.z ) << '\n' ;
				}
			}
		}

		util::optxCleanup() ;
	} catch( std::exception& e ) {
		std::cerr << "exception: " << e.what() << "\n" ;

		return 1 ;
	}

	return 0 ;
}
