// system includes
// none

// subsystem includes
// CUDA
#include <vector_types.h>

// local includes
#include "sphere.h"
#include "hoist.h"
#include "util.h"
#include "v.h"

// file specific includes
#include "scene.h"

using V::operator- ;
using V::operator* ;

void Scene::load() {
	things_.clear() ;

	Sphere* sphere = new Sphere( 1.f, 9 ) ;
	auto hoist = std::make_shared<Hoist>( sphere->vces(), sphere->ices() ) ;
	hoist->optics.type = Optics::TYPE_DIFFUSE ;
	hoist->optics.diffuse.albedo = { .5f, .5f, .5f } ;
	hoist->transform[0*4+0] =  1000.f ; // scale
	hoist->transform[1*4+1] =  1000.f ;
	hoist->transform[2*4+2] =  1000.f ;
	hoist->transform[0*4+3] =     0.f ; // translate
	hoist->transform[1*4+3] = -1000.f ;
	hoist->transform[2*4+3] =     0.f ;
	things_.push_back( hoist ) ;
	delete sphere ;

	for ( int a = -11 ; a<11 ; a++ ) {
		for ( int b = -11 ; b<11 ; b++ ) {
			auto select = util::rnd() ;
			const float3 center = { a+.9f*util::rnd(), .2f, b+.9f*util::rnd() } ;
			if ( V::len( center-make_float3( 4.f, .2f, 0.f ) )>.9f ) {
				if ( select<.8f ) {
					sphere = new Sphere() ;
					hoist = std::make_shared<Hoist>( sphere->vces(), sphere->ices() ) ;
					hoist->optics.type = Optics::TYPE_DIFFUSE ;
					hoist->optics.diffuse.albedo = V::rnd()*V::rnd() ;
					hoist->transform[0*4+0] = .2f ;
					hoist->transform[1*4+1] = .2f ;
					hoist->transform[2*4+2] = .2f ;
					hoist->transform[0*4+3] = center.x ;
					hoist->transform[1*4+3] = center.y ;
					hoist->transform[2*4+3] = center.z ;
					things_.push_back( hoist ) ;
					delete sphere ;
				} else if ( select<.95f ) {
					sphere = new Sphere() ;
					hoist = std::make_shared<Hoist>( sphere->vces(), sphere->ices() ) ;
					hoist->optics.type = Optics::TYPE_REFLECT ;
					hoist->optics.reflect.albedo = V::rnd( .5f, 1.f ) ;
					hoist->optics.reflect.fuzz = util::rnd( 0.f, .5f ) ;
					hoist->transform[0*4+0] = .2f ;
					hoist->transform[1*4+1] = .2f ;
					hoist->transform[2*4+2] = .2f ;
					hoist->transform[0*4+3] = center.x ;
					hoist->transform[1*4+3] = center.y ;
					hoist->transform[2*4+3] = center.z ;
					things_.push_back( hoist ) ;
					delete sphere ;
				} else {
					sphere = new Sphere( 1.f, 3 ) ;
					hoist = std::make_shared<Hoist>( sphere->vces(), sphere->ices() ) ;
					hoist->optics.type = Optics::TYPE_REFRACT ;
					hoist->optics.refract.index = 1.5f ;
					hoist->transform[0*4+0] = .2f ;
					hoist->transform[1*4+1] = .2f ;
					hoist->transform[2*4+2] = .2f ;
					hoist->transform[0*4+3] = center.x ;
					hoist->transform[1*4+3] = center.y ;
					hoist->transform[2*4+3] = center.z ;
					things_.push_back( hoist ) ;
					delete sphere ;
				}
			}
		}
	}


	sphere = new Sphere( 1.f, 8 ) ;
	hoist = std::make_shared<Hoist>( sphere->vces(), sphere->ices() ) ;
	hoist->optics.type = Optics::TYPE_REFRACT ;
	hoist->optics.refract.index  = 1.5f ;
	hoist->transform[0*4+0] = 1.f ;
	hoist->transform[1*4+1] = 1.f ;
	hoist->transform[2*4+2] = 1.f ;
	hoist->transform[0*4+3] = 0.f ;
	hoist->transform[1*4+3] = 1.f ;
	hoist->transform[2*4+3] = 0.f ;
	things_.push_back( hoist ) ;
	delete sphere ;

	sphere = new Sphere() ;
	hoist = std::make_shared<Hoist>( sphere->vces(), sphere->ices() ) ;
	hoist->optics.type = Optics::TYPE_DIFFUSE ;
	hoist->optics.diffuse.albedo = { .4f, .2f, .1f } ;
	hoist->transform[0*4+3] = -4.f ;
	hoist->transform[1*4+3] =  1.f ;
	hoist->transform[2*4+3] =  0.f ;
	things_.push_back( hoist ) ;
	delete sphere ;

	sphere = new Sphere( 1.f, 3 ) ;
	hoist = std::make_shared<Hoist>( sphere->vces(), sphere->ices() ) ;
	hoist->optics.type = Optics::TYPE_REFLECT ;
	hoist->optics.reflect.albedo = { .7f, .6f, .5f } ;
	hoist->optics.reflect.fuzz   = 0.f ;
	hoist->transform[0*4+3] = 4.f ;
	hoist->transform[1*4+3] = 1.f ;
	hoist->transform[2*4+3] = 0.f ;
	things_.push_back( hoist ) ;
	delete sphere ;
}

size_t Scene::size() const {
	return things_.size() ;
}
