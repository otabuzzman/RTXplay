#include <iostream>

#include "util.h"

#include "rgb.h"
#include "Things.h"
#include "Sphere.h"

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
	int c = 255 ; // color maximum value

	const auto ratio = 16./9. ;                       // aspect ratio
	const int w = 400;                                // image width in pixels
	const int h = static_cast<int>( w/ratio ) ;       // image height in pixels

	Things scene ;
	scene.add( make_shared<Sphere>(P( 0, 0, -1), .5 ) ) ;
	scene.add( make_shared<Sphere>(P( 0, -100.5, -1 ), 100) ) ;

	auto viewh = 2. ;                                 // virtual viewport height
	auto vieww = ratio*viewh ;                        // virtual viewport width
	auto focll = 1. ;                                 // focal length (projection point distance from ~plane)

	auto orig = P( 0, 0, 0 ) ;                        // camera origin
	auto hori = V( vieww, 0, 0 ) ;
	auto vert = V( 0, viewh, 0 ) ;
	auto bole = orig-hori/2-vert/2-V( 0, 0, focll ) ; // bottom left viewport corner

	std::cout
		<< "P3\n"	// magic PPM header
		<< w << ' ' << h << '\n' << c << '\n' ;

	for ( int j = h-1 ; j>=0 ; --j ) {
		for ( int i = 0 ; i<w ; ++i ) {
			auto u = (double) i/( w-1 ) ;
			auto v = (double) j/( h-1 ) ;

			Ray ray( orig, bole+u*hori+v*vert-orig ) ;
			C color = skies( ray, scene ) ;

			rgb( std::cout, color ) ;
		}
	}
}
