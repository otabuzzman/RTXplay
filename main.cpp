#include <iostream>

#include "util.h"

#include "Things.h"
#include "Sphere.h"
#include "Camera.h"
#include "Material.h"

const std::string rgbPP3( C color ) {
	char pp3[16] ;

	sprintf( pp3, "%d %d %d",
		static_cast<int>( 255*color.x() ),
		static_cast<int>( 255*color.y() ),
		static_cast<int>( 255*color.z() ) ) ;

	return std::string( pp3 ) ;

}

const std::string rgbPP3( C color, int spp ) {
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

double sphere( const P& center, double radius, const Ray& ray ) {
	V o = ray.ori()-center ;
	// auto a = dot( ray.dir(), ray.dir() ) ;
	auto a = ray.dir().len2() ;             // simplified: V dot V equals len(V)^2
	// auto b = 2.*dot( ray.dir(), o ) ;
	auto b = dot( ray.dir(), o ) ;          // simplified: b/2
	// auto c = dot( o, o )-radius*radius ;
	auto c = o.len2()-radius*radius ;       // simplified: V dot V equals len(V)^2
	// auto discriminant = b*b-4*a*c ;
	auto discriminant = b*b-a*c ;           // simplified: b/2

	if ( 0>discriminant )
		return -1. ;
	// else
	return ( -b-sqrt( discriminant ) )/a ;  // simplified: b/2
}

C trace( const Ray& ray, const Thing& scene ) {
	record rec ;
	if ( scene.hit( ray, 0, INF, rec ) )
		return .5*( rec.normal+C( 1, 1, 1 ) ) ;
	// else
	V unit = unitV( ray.dir() ) ;
	auto t = .5*( unit.y()+1. ) ;

	return ( 1.-t )*C( 1., 1., 1. )+t*C( .5, .7, 1. ) ;
}

C trace( const Ray& ray, const Thing& scene, int depth ) {
	record rec ; P s ;
	if ( 0>=depth )
		return C( 0, 0, 0 ) ;
	// else
	if ( scene.hit( ray, .001, INF, rec ) ) {
		Ray sprayed ;
		C attened ;
		if ( rec.m->spray( ray, rec, attened, sprayed ) )
			return attened*trace( sprayed, scene, depth-1 ) ;
		// else
		return C( 0, 0, 0 ) ;
	}
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
	const int dmax = 50 ;                                // recursion depth

	Things scene ;
	scene.add( make_shared<Sphere>( P( .0, -100.5, -1. ), 100.,  make_shared<Lambertian>( C( .8, .8, .0 ) ) ) ) ; // ground
	scene.add( make_shared<Sphere>( P( .0,     .0, -1. ),    .5, make_shared<Lambertian>( C( .1, .2, .5 ) ) ) ) ; // center
	scene.add( make_shared<Sphere>( P( -1.,    .0, -1. ),    .5, make_shared<Dielectric>( 1.5 ) ) ) ;             // l sphere
	scene.add( make_shared<Sphere>( P( -1.,    .0, -1. ),   -.4, make_shared<Dielectric>( 1.5 ) ) ) ;             // l sphere
	scene.add( make_shared<Sphere>( P( 1.,     .0, -1. ),    .5, make_shared<Metal>( C( .8, .6, .2 ), 1. ) ) ) ;  // r sphere

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
				color += trace( ray, scene, dmax ) ;
			}
			std::cout
				<< rgbPP3( color, spp ) << '\n' ;
		}
	}
}
