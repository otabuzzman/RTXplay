// system includes
// none

// subsystem includes
// CUDA
#include <vector_functions.h>
#include <vector_types.h>

// local includes
#include "sphere.h"
#include "thing.h"
#include "util.h"
#include "v.h"

// file specific includes
#include "scene.h"

using V::operator- ;
using V::operator* ;

void Scene::load() {
	things_.clear() ;

	auto sphere = std::make_shared<Sphere>( 1.f, 9 ) ;
	sphere->optics.type = Optics::TYPE_DIFFUSE ;
	sphere->optics.diffuse.albedo = { .5f, .5f, .5f } ;
	sphere->transform[0*4+0] =  1000.f ; // scale
	sphere->transform[1*4+1] =  1000.f ;
	sphere->transform[2*4+2] =  1000.f ;
	sphere->transform[0*4+3] =     0.f ; // translate
	sphere->transform[1*4+3] = -1000.f ;
	sphere->transform[2*4+3] =     0.f ;
	things_.push_back( sphere ) ;

	for ( int a = -11 ; a<11 ; a++ ) {
		for ( int b = -11 ; b<11 ; b++ ) {
			auto select = util::rnd() ;
			const float3 center = { a+.9f*util::rnd(), .2f, b+.9f*util::rnd() } ;
			if ( V::len( center-make_float3( 4.f, .2f, 0.f ) )>.9f ) {
				if ( select<.8f ) {
					sphere = std::make_shared<Sphere>() ;
					sphere->optics.type = Optics::TYPE_DIFFUSE ;
					sphere->optics.diffuse.albedo = V::rnd()*V::rnd() ;
					sphere->transform[0*4+0] = .2f ;
					sphere->transform[1*4+1] = .2f ;
					sphere->transform[2*4+2] = .2f ;
					sphere->transform[0*4+3] = center.x ;
					sphere->transform[1*4+3] = center.y ;
					sphere->transform[2*4+3] = center.z ;
					things_.push_back( sphere ) ;
				} else if ( select<.95f ) {
					sphere = std::make_shared<Sphere>() ;
					sphere->optics.type = Optics::TYPE_REFLECT ;
					sphere->optics.reflect.albedo = V::rnd( .5f, 1.f ) ;
					sphere->optics.reflect.fuzz = util::rnd( 0.f, .5f ) ;
					sphere->transform[0*4+0] = .2f ;
					sphere->transform[1*4+1] = .2f ;
					sphere->transform[2*4+2] = .2f ;
					sphere->transform[0*4+3] = center.x ;
					sphere->transform[1*4+3] = center.y ;
					sphere->transform[2*4+3] = center.z ;
					things_.push_back( sphere ) ;
				} else {
					sphere = std::make_shared<Sphere>( 1.f, 3 ) ;
					sphere->optics.type = Optics::TYPE_REFRACT ;
					sphere->optics.refract.index = 1.5f ;
					sphere->transform[0*4+0] = .2f ;
					sphere->transform[1*4+1] = .2f ;
					sphere->transform[2*4+2] = .2f ;
					sphere->transform[0*4+3] = center.x ;
					sphere->transform[1*4+3] = center.y ;
					sphere->transform[2*4+3] = center.z ;
					things_.push_back( sphere ) ;
				}
			}
		}
	}

	sphere = std::make_shared<Sphere>( 1.f, 8 ) ;
	sphere->optics.type = Optics::TYPE_REFRACT ;
	sphere->optics.refract.index  = 1.5f ;
	sphere->transform[0*4+0] = 1.f ;
	sphere->transform[1*4+1] = 1.f ;
	sphere->transform[2*4+2] = 1.f ;
	sphere->transform[0*4+3] = 0.f ;
	sphere->transform[1*4+3] = 1.f ;
	sphere->transform[2*4+3] = 0.f ;
	things_.push_back( sphere ) ;
	sphere = std::make_shared<Sphere>() ;
	sphere->optics.type = Optics::TYPE_DIFFUSE ;
	sphere->optics.diffuse.albedo = { .4f, .2f, .1f } ;
	sphere->transform[0*4+3] = -4.f ;
	sphere->transform[1*4+3] =  1.f ;
	sphere->transform[2*4+3] =  0.f ;
	things_.push_back( sphere ) ;
	sphere = std::make_shared<Sphere>( 1.f, 3 ) ;
	sphere->optics.type = Optics::TYPE_REFLECT ;
	sphere->optics.reflect.albedo = { .7f, .6f, .5f } ;
	sphere->optics.reflect.fuzz   = 0.f ;
	sphere->transform[0*4+3] = 4.f ;
	sphere->transform[1*4+3] = 1.f ;
	sphere->transform[2*4+3] = 0.f ;
	things_.push_back( sphere ) ;
}

size_t Scene::size() const {
	return things_.size() ;
}
