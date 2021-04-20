#include "camera.h"
#include "optics.h"
#include "sphere.h"
#include "things.h"

const std::string sRGB( const C color ) {
	char pp3[16] ;

	sprintf( pp3, "%d %d %d",
		static_cast<int>( 255*color.x() ),
		static_cast<int>( 255*color.y() ),
		static_cast<int>( 255*color.z() ) ) ;

	return std::string( pp3 ) ;
}

const std::string sRGB( const C color, int spp ) {
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

const C trace( const Ray& ray, const Thing& scene ) {
	Binding binding ;
	if ( scene.hit( ray, 0, kInfinity, binding ) )
		return .5*( binding.normal+C( 1, 1, 1 ) ) ;

	V unit = unitV( ray.dir() ) ;
	auto t = .5*( unit.y()+1. ) ;

	return ( 1.-t )*C( 1., 1., 1. )+t*C( .5, .7, 1. ) ;
}

const C trace( const Ray& ray, const Thing& scene, int depth ) {
	Binding binding ;
	if ( scene.hit( ray, kAcne0, kInfinity, binding ) ) {
		Ray sprayed ;
		C attened ;
		if ( depth>0 && binding.optics->spray( ray, binding, attened, sprayed ) )
			return attened*trace( sprayed, scene, depth-1 ) ;

		return C( 0, 0, 0 ) ;
	}

	V unit = unitV( ray.dir() ) ;
	auto t = .5*( unit.y()+1. ) ;

	return ( 1.-t )*C( 1., 1., 1. )+t*C( .5, .7, 1. ) ;
}

const Things scene() {
	Things s ;

	s.add( std::make_shared<Sphere>( P( 0, -1000, 0 ), 1000., std::make_shared<Diffuse>( C( .5, .5, .5 ) ) ) ) ;

	for ( int a = -11 ; a<11 ; a++ ) {
		for ( int b = -11 ; b<11 ; b++ ) {
			auto select = rnd() ;
			P center( a+.9*rnd(), .2, b+.9*rnd() ) ;
			if ( ( center-P( 4, .2, 0 ) ).len()>.9 ) {
				if ( select<.8 ) {
					auto albedo = C::rnd()*C::rnd() ;
					s.add( std::make_shared<Sphere>( center, .2, std::make_shared<Diffuse>( albedo ) ) ) ;
				} else if ( select<.95 ) {
					auto albedo = C::rnd( .5, 1. ) ;
					auto fuzz = rnd( 0, .5 ) ;
					s.add( std::make_shared<Sphere>( center, .2, std::make_shared<Reflect>( albedo, fuzz ) ) ) ;
				} else {
					s.add( std::make_shared<Sphere>( center, .2, std::make_shared<Refract>( 1.5 ) ) ) ;
				}
			}
		}
	}

	s.add( std::make_shared<Sphere>( P(  0, 1, 0 ), 1., std::make_shared<Refract>( 1.5 ) ) ) ;
	s.add( std::make_shared<Sphere>( P( -4, 1, 0 ), 1., std::make_shared<Diffuse>( C( .4, .2, .1 ) ) ) ) ;
	s.add( std::make_shared<Sphere>( P(  4, 1, 0 ), 1., std::make_shared<Reflect>( C( .7, .6, .5 ), 0 ) ) ) ;

	return s ;
}

int main() {
	Things things = ::scene() ;

	double aspratio = 3./2. ;

	P eye( 13, 2, 3 ) ;
	P pat( 0, 0, 0 ) ;
	V vup( 0, 1, 0 ) ;
	double aperture = .1 ;
	double distance = 10. ;

	Camera camera ;
	camera.set( eye, pat, vup, 20., aspratio, aperture, distance ) ;

	const int w = 1200 ;                           // image width in pixels
	const int h = static_cast<int>( w/aspratio ) ; // image height in pixels
	const int spp = 10 ;                           // samples per pixel
	const int depth = 50 ;                         // recursion depth

	std::cout
		<< "P3\n"	// magic PPM header
		<< w << ' ' << h << '\n' << 255 << '\n' ;

	for ( int y = h-1 ; y>=0 ; --y ) {
		std::cerr << "\r" << y << ' ' << std::flush ;
		for ( int x = 0 ; x<w ; ++x ) {
			C color( 0, 0, 0 ) ;
			for ( int k = 0 ; k<spp ; ++k ) {
				// transform x/y pixel ccord (range 0/0 to w/h)
				// into s/t viewport coords (range -1/-1 to 1/1)
				auto s = 2.*( x+rnd() )/( w-1 )-1. ;
				auto t = 2.*( y+rnd() )/( h-1 )-1. ;

				Ray ray = camera.ray( s, t ) ;
				color += trace( ray, things, depth ) ;
			}
			std::cout
				<< sRGB( color, spp ) << '\n' ;
		}
	}
}
