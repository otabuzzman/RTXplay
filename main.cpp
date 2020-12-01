#include <iostream>

#include "util.h"

#include "rgb.h"
#include "Things.h"
#include "Sphere.h"
#include "Camera.h"

double sphere( const P& center, double radius, const Ray& ray ) {
	V o = ray.ori()-center ;
	// auto a = dot( ray.dir(), ray.dir() ) ;
	auto a = ray.dir().plen() ;             // simplified: V dot V equals len(V)^2
	// auto b = 2.*dot( ray.dir(), o ) ;
	auto b = dot( ray.dir(), o ) ;          // simplified: b/2
	// auto c = dot( o, o )-radius*radius ;
	auto c = o.plen()-radius*radius ; // simplified: V dot V equals len(V)^2
	// auto discriminant = b*b-4*a*c ;
	auto discriminant = b*b-a*c ;           // simplified: b/2

	if ( 0>discriminant )
		return -1. ;
	// else
	return ( -b-sqrt( discriminant ) )/a ;  // simplified: b/2
}

C skies( const Ray& ray, const Thing& scene ) {
	record rec ;
	if ( scene.hit( ray, 0, INF, rec ) )
		return .5*( rec.normal+C( 1, 1, 1 ) ) ;
	// else
	V unit = unitV( ray.dir() ) ;
	auto t = .5*( unit.y()+1. ) ;

	return ( 1.-t )*C( 1., 1., 1. )+t*C( .5, .7, 1. ) ;
}

int main() {
	Camera camera ;

	const int w = 400;                                   // image width in pixels
	const int h = static_cast<int>( w/camera.ratio() ) ; // image height in pixels
	const int spp = 100 ;                                // samples per pixel

	Things scene ;
	scene.add( make_shared<Sphere>(P( 0, 0, -1), .5 ) ) ;
	scene.add( make_shared<Sphere>(P( 0, -100.5, -1 ), 100) ) ;

	std::cout
		<< "P3\n"	// magic PPM header
		<< w << ' ' << h << '\n' << 255 << '\n' ;

	for ( int j = h-1 ; j>=0 ; --j ) {
		for ( int i = 0 ; i<w ; ++i ) {
			C color( 0, 0, 0 ) ;
			for ( int s = 0 ; s<spp ; ++s ) {
				auto u = ( i+rnd() )/(w-1) ;
				auto v = ( j+rnd() )/(h-1) ;

				Ray ray = camera.ray( u, v ) ;
				color += skies( ray, scene ) ;
			}
			rgb( std::cout, color, spp ) ;
		}
	}
}
