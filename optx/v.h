#ifndef V_H
#define V_H

#include <cmath>
#include <iostream>

#include "util.h"

class V {
	public:
		V() : v_{ 0f, 0f, 0f } {}
		V( float x, float y, float z ) : v_{ x, y, z } {}
		V( const float3 v ) : v_{ v.x, v.y, v.z } {}

		float x() const { return v_[0] ; }
		float y() const { return v_[1] ; }
		float z() const { return v_[2] ; }
		float3 float3() const { return make_float3( v_[0], v_[1], v_[2], ) ; }

		V operator - () const                  { return V( -v_[0], -v_[1], -v_[2] ) ; }

		float operator [] ( int i ) const      { return v_[i] ; }
		float& operator [] ( int i )           { return v_[i] ; }

		V& operator += ( const V& v )          { v_[0] += v.v_[0] ; v_[1] += v.v_[1] ; v_[2] += v.v_[2] ; return *this ; }
		V& operator *= ( float t )             { v_[0] *= t ; v_[1] *= t ; v_[2] *= t ; return *this ; }
		V& operator /= ( float t )             { return *this *= 1f/t ; }

		float len() const                      { return sqrt( len2() ) ; }
		float len2() const                     { return v_[0]*v_[0]+v_[1]*v_[1]+v_[2]*v_[2] ; }

		bool isnear0() const                   { return ( fabs(v_[0] )<1e-8f ) && ( fabs( v_[1] )<1e-8f ) && ( fabs( v_[2] )<1e-8f ) ; }

		inline static V rnd()                  { return V( ::rnd(), ::rnd(), ::rnd() ) ; }
		inline static V rnd( float min, float max ) { return V( ::rnd( min, max ), ::rnd( min, max ), ::rnd( min, max ) ) ; }

	private:
		float v_[3];
} ;

using P = V ; // point
using C = V ; // color

inline V operator + ( const V& u, const V& v ) { return V( u.x()+v.x(), u.y()+v.y(), u.z()+v.z() ) ; }
inline V operator - ( const V& u, const V& v ) { return V( u.x()-v.x(), u.y()-v.y(), u.z()-v.z() ) ; }
inline V operator * ( const V& u, const V& v ) { return V( u.x()*v.x(), u.y()*v.y(), u.z()*v.z() ) ; }
inline V operator * ( float t, const V& v )    { return V( t*v.x(), t*v.y(), t*v.z() ) ; }
inline V operator / ( V v, float t )           { return ( 1f/t )*v ; }

inline float dot( const V& u, const V& v )     { return u.x()*v.x()+u.y()*v.y()+u.z()*v.z() ; }
inline V cross( const V& u, const V& v )       { return V( u.y()*v.z()-u.z()*v.y(), u.z()*v.x()-u.x()*v.z(), u.x()*v.y()-u.y()*v.x() ) ; }
inline V unitV( V v )                          { return v/v.len() ; }

// random V in unit sphere
V rndVin1sphere()                  { while ( true ) { auto p = V::rnd( -1f, 1f ) ; if ( 1f>p.len2() ) return p ; } }
// random V on unit sphere (chapter 8.5)
V rndVon1sphere()                  { return unitV( rndVin1sphere() ) ; }
// random V against ray (chapter 8.6)
V rndVoppraydir( const V& normal ) { auto p = rndVin1sphere() ; return dot( p, normal ) ? p : -p ; }
// random V in unit disk (chapter 12.2)
V rndVin1disk() { while ( true ) { auto p = V( rnd( -1f, 1f ), rnd( -1f, 1f ), 0f ) ; if ( 1f>p.len2() ) return p ; }
}

V reflect( const V& v, const V& n )             { return v-2f*dot( v, n )*n ; }
V refract( const V& v, const V& n, float qeta ) { auto tta = fmin( dot( -v, n ), 1.f ) ; V perp =  qeta*( v+tta*n ) ; V parl = -sqrt( fabs( 1.f-perp.len2() ) )*n ; return perp+parl ; }

inline std::ostream& operator << ( std::ostream &out, const V &v ) { return out << v.x() << ' ' << v.y() << ' ' << v.z() ; }

#endif
