#ifndef SPHERE_H
#define SPHERE_H

#include <vector>

#include "v.h"

class Sphere : public Triangles {
	public:
		Sphere() {}
		Sphere( const P center, float radius ) : center_( center ), radius_( radius ) { icosahedron( 4 ) }

		P     center() const { return center_ ; }
		float radius() const { return radius_ ; }

	private:
		P     center_ ;
		float radius_ ;

		std::vector<V> triangles_ ;

		void icosahedron( int number_of_subdivisions ) const {
			// https://rechneronline.de/pi/icosahedron.php
			// r = a/4*sqrt(10+2*sqrt(5)) | r = 1
			// a = 1/sqrt(10+2*sqrt(5))*4
			// b = a/4*(1+sqrt(5))
			// precalculated values for unit icosahedron
			float a = 1.05146222424f/2f ; // half edge length
			float b = 0.52573111211f    ; // midsphere radius

			// icosahedron's 12 vertices
			auto v00 = P(  0,  b, -a ) ;
			auto v01 = P(  b,  a,  0 ) ;
			auto v02 = P( -b,  a,  0 ) ;
			auto v03 = P(  0,  b,  a ) ;
			auto v04 = P(  0, -b,  a ) ;
			auto v05 = P( -a,  0,  b ) ;
			auto v06 = P(  0, -b, -a ) ;
			auto v07 = P(  a,  0, -b ) ;
			auto v08 = P(  a,  0,  b ) ;
			auto v09 = P( -a,  0, -b ) ;
			auto v10 = P(  b, -a,  0 ) ;
			auto v11 = P( -b, -a,  0 ) ;

			// icosahedron's 20 triangles
			divide( v00, v01, v02, number_of_subdivisions ) ;
			divide( v03, v02, v01, number_of_subdivisions ) ;
			divide( v03, v04, v05, number_of_subdivisions ) ;
			divide( v03, v08, v04, number_of_subdivisions ) ;
			divide( v00, v06, v07, number_of_subdivisions ) ;
			divide( v00, v09, v06, number_of_subdivisions ) ;
			divide( v04, v10, v11, number_of_subdivisions ) ;
			divide( v06, v11, v10, number_of_subdivisions ) ;
			divide( v02, v05, v09, number_of_subdivisions ) ;
			divide( v11, v09, v05, number_of_subdivisions ) ;
			divide( v01, v07, v08, number_of_subdivisions ) ;
			divide( v10, v08, v07, number_of_subdivisions ) ;
			divide( v03, v05, v02, number_of_subdivisions ) ;
			divide( v03, v01, v08, number_of_subdivisions ) ;
			divide( v00, v02, v09, number_of_subdivisions ) ;
			divide( v00, v07, v01, number_of_subdivisions ) ;
			divide( v06, v09, v11, number_of_subdivisions ) ;
			divide( v06, v10, v07, number_of_subdivisions ) ;
			divide( v04, v11, v05, number_of_subdivisions ) ;
			divide( v04, v08, v10, number_of_subdivisions ) ;
		}

		void divide( const P& a, const P& b, const P& c, int pending_subdivisions ) const {
			if  ( pending_subdivisions>0 ) {
				auto ab = ( .5*( a+b ) ).unitV() ;
				auto bc = ( .5*( b+c ) ).unitV() ;
				auto ca = ( .5*( c+a ) ).unitV() ;

				divide (  a, ab, ca, pending_subdivisions-1 ) ;
				divide ( ab,  b, bc, pending_subdivisions-1 ) ;
				divide ( ca, bc,  c, pending_subdivisions-1 ) ;
				divide ( ab, bc, ca, pending_subdivisions-1 ) ;
			} else {
				triangles_.add( V( center_+radius_*A ) ;
				triangles_.add( V( center_+radius_*B ) ;
				triangles_.add( V( center_+radius_*C ) ;
			}
		}
} ;

#endif
