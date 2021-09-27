#include <vector_functions.h>
#include <vector_types.h>

#include "optics.h"
#include "sphere.h"
#include "util.h"
#include "v.h"

#include "scene.h"

using V::operator- ;
using V::operator* ;

void Scene::load() {
	Optics o ;

	o.type = OPTICS_TYPE_DIFFUSE ;
	o.diffuse.albedo = { .5f, .5f, .5f } ;
	things_.push_back( std::make_shared<Sphere>( make_float3( 0.f, -1000.f, 0.f ), 1000.f, o, false, 9 ) ) ;

	for ( int a = -11 ; a<11 ; a++ ) {
		for ( int b = -11 ; b<11 ; b++ ) {
			auto bbox = false ; // .3f>util::rnd() ? true : false ;
			auto select = util::rnd() ;
			float3 center = { a+.9f*util::rnd(), .2f, b+.9f*util::rnd() } ;
			if ( V::len( center-make_float3( 4.f, .2f, 0.f ) )>.9f ) {
				if ( select<.8f ) {
					o.type = OPTICS_TYPE_DIFFUSE ;
					o.diffuse.albedo = V::rnd()*V::rnd() ;
					things_.push_back( std::make_shared<Sphere>( center, .2f, o, bbox ) ) ;
				} else if ( select<.95f ) {
					o.type = OPTICS_TYPE_REFLECT ;
					o.reflect.albedo = V::rnd( .5f, 1.f ) ;
					o.reflect.fuzz = util::rnd( 0.f, .5f ) ;
					things_.push_back( std::make_shared<Sphere>( center, .2f, o, bbox ) ) ;
				} else {
					o.type = OPTICS_TYPE_REFRACT ;
					o.refract.index = 1.5f ;
					things_.push_back( std::make_shared<Sphere>( center, .2f, o, bbox, 3 ) ) ;
				}
			}
		}
	}

	o.type = OPTICS_TYPE_REFRACT ;
	o.refract.index  = 1.5f ;
	things_.push_back( std::make_shared<Sphere>( make_float3(  0.f, 1.f, 0.f ), 1.f, o, false, 8 ) ) ;
	o.type = OPTICS_TYPE_DIFFUSE ;
	o.diffuse.albedo = { .4f, .2f, .1f } ;
	things_.push_back( std::make_shared<Sphere>( make_float3( -4.f, 1.f, 0.f ), 1.f, o ) ) ;
	o.type = OPTICS_TYPE_REFLECT ;
	o.reflect.albedo = { .7f, .6f, .5f } ;
	o.reflect.fuzz   = 0.f ;
	things_.push_back( std::make_shared<Sphere>( make_float3(  4.f, 1.f, 0.f ), 1.f, o, false, 3 ) ) ;
}

size_t Scene::size() const {
	return things_.size() ;
}
