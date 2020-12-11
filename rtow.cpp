#include <iostream>
#include <string>

#include "util.h"

#include "things.h"
#include "sphere.h"
#include "camera.h"
#include "optics.h"

const std::string rgbPP3( const C color ) {
	char pp3[16] ;

	sprintf( pp3, "%d %d %d",
		static_cast<int>( 255*color.x() ),
		static_cast<int>( 255*color.y() ),
		static_cast<int>( 255*color.z() ) ) ;

	return std::string( pp3 ) ;
}

const std::string rgbPP3( const C color, int spp ) {
	char pp3[16] ;
	auto r = color.x() ;
	auto g = color.y() ;
	auto b = color.z() ;

	auto s = 1./spp ;
	r = sqrt( s*r ) ; g = sqrt( s*g ) ; b = sqrt( s*b ) ;

	sprintf( pp3, "%d %d %d",
		static_cast<int>( 256*clamp( r, 0, .999 ) ),
		static_cast<int>( 256*clamp( g, 0, .999 ) ),
		static_cast<int>( 256*clamp( b, 0, .999 ) ) ) ;

	return std::string( pp3 ) ;
}

C trace( const Ray& ray, const Thing& scene ) {
	Bucket bucket ;
	if ( scene.hit( ray, 0, kInfinty, bucket ) )
		return .5*( bucket.normal+C( 1, 1, 1 ) ) ;
	// else
	V unit = unitV( ray.dir() ) ;
	auto t = .5*( unit.y()+1. ) ;

	return ( 1.-t )*C( 1., 1., 1. )+t*C( .5, .7, 1. ) ;
}

C trace( const Ray& ray, const Thing& scene, int depth ) {
	Bucket bucket ;
	if ( scene.hit( ray, .001, kInfinty, bucket ) ) {
		Ray sprayed ;
		C attened ;
		if ( depth>0 && bucket.optics->spray( ray, bucket, attened, sprayed ) )
			return attened*trace( sprayed, scene, depth-1 ) ;
		// else
		return C( 0, 0, 0 ) ;
	}
	// else
	V unit = unitV( ray.dir() ) ;
	auto t = .5*( unit.y()+1. ) ;

	return ( 1.-t )*C( 1., 1., 1. )+t*C( .5, .7, 1. ) ;
}

Things scene() {
	Things s ;

	s.add( make_shared<Sphere>( P( 0, -1000, 0 ), 1000., make_shared<Diffuse>( C( .5, .5, .5 ) ) ) ) ;

	for ( int a = -11 ; a<11 ; a++ ) {
		for ( int b = -11 ; b<11; b++ ) {
			auto select = rnd() ;
			P center( a+.9*rnd(), .2, b+.9*rnd() ) ;
			if ( ( center-P( 4, .2, 0 ) ).len()>.9 ) {
				if ( select<.8 ) {
					auto albedo = C::rnd()*C::rnd() ;
					s.add( make_shared<Sphere>( center, .2, make_shared<Diffuse>( albedo ) ) ) ;
				} else if ( select<.95 ) {
					auto albedo = C::rnd( .5, 1. ) ;
					auto fuzz = rnd( 0, .5 ) ;
					s.add( make_shared<Sphere>( center, .2, make_shared<Reflect>( albedo, fuzz ) ) ) ;
				} else {
					s.add( make_shared<Sphere>( center, .2, make_shared<Refract>( 1.5 ) ) ) ;
				}
			}
		}
	}

	s.add( make_shared<Sphere>( P(  0, 1, 0 ), 1., make_shared<Refract>( 1.5 ) ) ) ;
	s.add( make_shared<Sphere>( P( -4, 1, 0 ), 1., make_shared<Diffuse>( C( .4, .2, .1 ) ) ) ) ;
	s.add( make_shared<Sphere>( P(  4, 1, 0 ), 1., make_shared<Reflect>( C( .7, .6, .5 ), 0 ) ) ) ;

	return s ;
}

int main() {
	Things scene = ::scene() ;

	double aspect_ratio = 3./2. ;

	P pos( 13, 2, 3 ) ;
	P pov( 0, 0, 0 ) ;
	V up( 0, 1, 0 ) ;
	double aperture = .1 ;
	double distance = 10. ;

	Camera camera( pos, pov, up, 20., aspect_ratio, aperture, distance ) ;

	const int w = 1200 ;                               // image width in pixels
	const int h = static_cast<int>( w/aspect_ratio ) ; // image height in pixels
	const int spp = 50 ;                               // samples per pixel
	const int depth = 50 ;                             // recursion depth

	std::cout
		<< "P3\n"	// magic PPM header
		<< w << ' ' << h << '\n' << 255 << '\n' ;

	for ( int j = h-1 ; j>=0 ; --j ) {
		std::cerr << "\r" << j << ' ' << std::flush ;
		for ( int i = 0 ; i<w ; ++i ) {
			C color( 0, 0, 0 ) ;
			for ( int k = 0 ; k<spp ; ++k ) {
				auto s = ( i+rnd() )/( w-1 ) ;
				auto t = ( j+rnd() )/( h-1 ) ;

				Ray ray = camera.ray( s, t ) ;
				color += trace( ray, scene, depth ) ;
			}
			std::cout
				<< rgbPP3( color, spp ) << '\n' ;
		}
	}
}
