// system includes
// none

// subsystem includes
// CUDA
#include <vector_types.h>

// local includes
#include "args.h"
#include "object.h"
#include "util.h"
#include "v.h"

// file specific includes
#include "scene.h"

// common globals
extern Args* args ;

using V::operator- ;
using V::operator* ;

void Scene::load( size_t* num_utm ) {
	things_.clear() ;

	// create unique triangle meshes (UTM)
	auto sphere_3 = std::make_shared<Object>( "sphere_3.scn" ) ;
	auto sphere_6 = std::make_shared<Object>( "sphere_6.scn" ) ;
	auto sphere_8 = std::make_shared<Object>( "sphere_8.scn" ) ;
	auto sphere_9 = std::make_shared<Object>( "sphere_9.scn" ) ;

	if ( args->flag_S() ) {
		fprintf( stderr, "sphere_3.scn: %u triangles %u vertices\n", static_cast<unsigned int>( sphere_3->vces().size() ), static_cast<unsigned int>( sphere_3->ices().size() ) ) ;
		fprintf( stderr, "sphere_6.scn: %u triangles %u vertices\n", static_cast<unsigned int>( sphere_6->vces().size() ), static_cast<unsigned int>( sphere_6->ices().size() ) ) ;
		fprintf( stderr, "sphere_8.scn: %u triangles %u vertices\n", static_cast<unsigned int>( sphere_8->vces().size() ), static_cast<unsigned int>( sphere_8->ices().size() ) ) ;
		fprintf( stderr, "sphere_9.scn: %u triangles %u vertices\n", static_cast<unsigned int>( sphere_9->vces().size() ), static_cast<unsigned int>( sphere_9->ices().size() ) ) ;
	}

	meshes_.push_back( sphere_3 ) ;
	meshes_.push_back( sphere_6 ) ;
	meshes_.push_back( sphere_8 ) ;
	meshes_.push_back( sphere_9 ) ;

	sphere_9->optics.type = Optics::TYPE_DIFFUSE ;
	sphere_9->optics.diffuse.albedo = { .5f, .5f, .5f } ;
	sphere_9->transform[0*4+0] =  1000.f ; // scale
	sphere_9->transform[1*4+1] =  1000.f ;
	sphere_9->transform[2*4+2] =  1000.f ;
	sphere_9->transform[0*4+3] =     0.f ; // translate
	sphere_9->transform[1*4+3] = -1000.f ;
	sphere_9->transform[2*4+3] =     0.f ;
	things_.push_back( *sphere_9 ) ;

	for ( int a = -11 ; a<11 ; a++ ) {
		for ( int b = -11 ; b<11 ; b++ ) {
			auto select = util::rnd() ;
			const float3 center = { a+.9f*util::rnd(), .2f, b+.9f*util::rnd() } ;
			if ( V::len( center-make_float3( 4.f, .2f, 0.f ) )>.9f ) {
				if ( select<.8f ) {
					sphere_6->optics.type = Optics::TYPE_DIFFUSE ;
					sphere_6->optics.diffuse.albedo = V::rnd()*V::rnd() ;
					sphere_6->transform[0*4+0] = .2f ;
					sphere_6->transform[1*4+1] = .2f ;
					sphere_6->transform[2*4+2] = .2f ;
					sphere_6->transform[0*4+3] = center.x ;
					sphere_6->transform[1*4+3] = center.y ;
					sphere_6->transform[2*4+3] = center.z ;
					things_.push_back( *sphere_6 ) ;
				} else if ( select<.95f ) {
					sphere_6->optics.type = Optics::TYPE_REFLECT ;
					sphere_6->optics.reflect.albedo = V::rnd( .5f, 1.f ) ;
					sphere_6->optics.reflect.fuzz = util::rnd( 0.f, .5f ) ;
					sphere_6->transform[0*4+0] = .2f ;
					sphere_6->transform[1*4+1] = .2f ;
					sphere_6->transform[2*4+2] = .2f ;
					sphere_6->transform[0*4+3] = center.x ;
					sphere_6->transform[1*4+3] = center.y ;
					sphere_6->transform[2*4+3] = center.z ;
					things_.push_back( *sphere_6 ) ;
				} else {
					sphere_3->optics.type = Optics::TYPE_REFRACT ;
					sphere_3->optics.refract.index = 1.5f ;
					sphere_3->transform[0*4+0] = .2f ;
					sphere_3->transform[1*4+1] = .2f ;
					sphere_3->transform[2*4+2] = .2f ;
					sphere_3->transform[0*4+3] = center.x ;
					sphere_3->transform[1*4+3] = center.y ;
					sphere_3->transform[2*4+3] = center.z ;
					things_.push_back( *sphere_3 ) ;
				}
			}
		}
	}

	sphere_8->optics.type = Optics::TYPE_REFRACT ;
	sphere_8->optics.refract.index  = 1.5f ;
	sphere_8->transform[0*4+0] = 1.f ;
	sphere_8->transform[1*4+1] = 1.f ;
	sphere_8->transform[2*4+2] = 1.f ;
	sphere_8->transform[0*4+3] = 0.f ;
	sphere_8->transform[1*4+3] = 1.f ;
	sphere_8->transform[2*4+3] = 0.f ;
	things_.push_back( *sphere_8 ) ;

	sphere_6->optics.type = Optics::TYPE_DIFFUSE ;
	sphere_6->optics.diffuse.albedo = { .4f, .2f, .1f } ;
	sphere_6->transform[0*4+0] = 1.f ;
	sphere_6->transform[1*4+1] = 1.f ;
	sphere_6->transform[2*4+2] = 1.f ;
	sphere_6->transform[0*4+3] = -4.f ;
	sphere_6->transform[1*4+3] =  1.f ;
	sphere_6->transform[2*4+3] =  0.f ;
	things_.push_back( *sphere_6 ) ;

	sphere_3->optics.type = Optics::TYPE_REFLECT ;
	sphere_3->optics.reflect.albedo = { .7f, .6f, .5f } ;
	sphere_3->optics.reflect.fuzz   = 0.f ;
	sphere_3->transform[0*4+0] = 1.f ;
	sphere_3->transform[1*4+1] = 1.f ;
	sphere_3->transform[2*4+2] = 1.f ;
	sphere_3->transform[0*4+3] = 4.f ;
	sphere_3->transform[1*4+3] = 1.f ;
	sphere_3->transform[2*4+3] = 0.f ;
	things_.push_back( *sphere_3 ) ;

	if ( num_utm )
		*num_utm = 4 ; // return number of UTMs
}
