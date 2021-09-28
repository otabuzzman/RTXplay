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
	Optics optics ;
	float  matrix[12] = { // identity default
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 0
	} ;
	std::shared_ptr<Sphere> sphere ;

	things_.clear() ;

	optics.type = Optics::TYPE_DIFFUSE ;
	optics.diffuse.albedo = { .5f, .5f, .5f } ;
	matrix[0*3+0] =  1000.f ; // scale
	matrix[1*3+1] =  1000.f ;
	matrix[2*3+2] =  1000.f ;
	matrix[3*3+0] =     0.f ; // translate
	matrix[3*3+1] = -1000.f ;
	matrix[3*3+2] =     0.f ;
	sphere = std::make_shared<Sphere>( 1.f, optics, false, 9 ) ;
	sphere->transform( matrix ) ;
	things_.push_back( sphere ) ;

	for ( int a = -11 ; a<11 ; a++ ) {
		for ( int b = -11 ; b<11 ; b++ ) {
			auto bbox = false ; // .3f>util::rnd() ? true : false ;
			auto select = util::rnd() ;
			const float3 center = { a+.9f*util::rnd(), .2f, b+.9f*util::rnd() } ;
			matrix[0*3+0] = .2f ;
			matrix[1*3+1] = .2f ;
			matrix[2*3+2] = .2f ;
			matrix[3*3+0] = center.x ;
			matrix[3*3+1] = center.y ;
			matrix[3*3+2] = center.z ;
			if ( V::len( center-make_float3( 4.f, .2f, 0.f ) )>.9f ) {
				if ( select<.8f ) {
					optics.type = Optics::TYPE_DIFFUSE ;
					optics.diffuse.albedo = V::rnd()*V::rnd() ;
					sphere = std::make_shared<Sphere>( 1.f, optics, bbox ) ;
					sphere->transform( matrix ) ;
					things_.push_back( sphere ) ;
				} else if ( select<.95f ) {
					optics.type = Optics::TYPE_REFLECT ;
					optics.reflect.albedo = V::rnd( .5f, 1.f ) ;
					optics.reflect.fuzz = util::rnd( 0.f, .5f ) ;
					sphere = std::make_shared<Sphere>( 1.f, optics, bbox ) ;
					sphere->transform( matrix ) ;
					things_.push_back( sphere ) ;
				} else {
					optics.type = Optics::TYPE_REFRACT ;
					optics.refract.index = 1.5f ;
					sphere = std::make_shared<Sphere>( 1.f, optics, bbox, 3 ) ;
					sphere->transform( matrix ) ;
					things_.push_back( sphere ) ;
				}
			}
		}
	}

	optics.type = Optics::TYPE_REFRACT ;
	optics.refract.index  = 1.5f ;
	matrix[0*3+0] = 1.f ;
	matrix[1*3+1] = 1.f ;
	matrix[2*3+2] = 1.f ;
	matrix[3*3+0] = 0.f ;
	matrix[3*3+1] = 1.f ;
	matrix[3*3+2] = 0.f ;
	sphere = std::make_shared<Sphere>( 1.f, optics, false, 8 ) ;
	sphere->transform( matrix ) ;
	things_.push_back( sphere ) ;
	optics.type = Optics::TYPE_DIFFUSE ;
	optics.diffuse.albedo = { .4f, .2f, .1f } ;
	matrix[3*3+0] = -4.f ;
	matrix[3*3+1] =  1.f ;
	matrix[3*3+2] =  0.f ;
	sphere = std::make_shared<Sphere>( 1.f, optics ) ;
	sphere->transform( matrix ) ;
	things_.push_back( sphere ) ;
	optics.type = Optics::TYPE_REFLECT ;
	optics.reflect.albedo = { .7f, .6f, .5f } ;
	optics.reflect.fuzz   = 0.f ;
	matrix[3*3+0] = 4.f ;
	matrix[3*3+1] = 1.f ;
	matrix[3*3+2] = 0.f ;
	sphere = std::make_shared<Sphere>( 1.f, optics, false, 3 ) ;
	sphere->transform( matrix ) ;
	things_.push_back( sphere ) ;
}

size_t Scene::size() const {
	return things_.size() ;
}
