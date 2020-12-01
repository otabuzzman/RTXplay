#include <iostream>

#include "V.h"
#include "rgb.h"
#include "Ray.h"

C skies( const Ray& ray ) {
	V unit = unitV( ray.dir() ) ;
	auto t = .5*( unit.y()+1. ) ;

	return ( 1.-t )*C( 1., 1., 1. )+t*C( .5, .7, 1. ) ;
}

int main() {
	int c = 255 ; // color maximum value

	const auto ratio = 16./9. ;                       // aspect ratio
	const int w = 400;                                // image width in pixels
	const int h = static_cast<int>( w/ratio ) ;       // image height in pixels

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
			C color = skies( ray ) ;

			rgb( std::cout, color ) ;
		}
	}
}
