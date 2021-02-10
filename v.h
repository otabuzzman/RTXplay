#ifndef V_H
#define V_H

#include <cmath>
#include <iostream>

#include "util.h"

class V {
	public:
		V() : v_{ 0, 0, 0 } {}
		V( double x, double y, double z ) : v_{ x, y, z } {}

		double x() const { return v_[0] ; }
		double y() const { return v_[1] ; }
		double z() const { return v_[2] ; }

		V operator - () const                  { return V( -v_[0], -v_[1], -v_[2] ) ; }

		double operator [] ( int i ) const     { return v_[i] ; }
		double& operator [] ( int i )          { return v_[i] ; }

		V& operator += ( const V& v )          { v_[0] += v.v_[0] ; v_[1] += v.v_[1] ; v_[2] += v.v_[2] ; return *this ; }
		V& operator *= ( double t )            { v_[0] *= t ; v_[1] *= t ; v_[2] *= t ; return *this ; }
		V& operator /= ( double t )            { return *this *= 1/t ; }

		double len() const                     { return sqrt( dot() ) ; }
		double dot() const                     { return v_[0]*v_[0]+v_[1]*v_[1]+v_[2]*v_[2] ; }

		bool isnear0() const                   { return ( fabs(v_[0] )<kNear0 ) && ( fabs( v_[1] )<kNear0 ) && ( fabs( v_[2] )<kNear0 ) ; }

		inline static V rnd()                  { return V( ::rnd(), ::rnd(), ::rnd() ) ; }
		inline static V rnd( double min, double max ) { return V( ::rnd( min, max ), ::rnd( min, max ), ::rnd( min, max ) ) ; }

	private:
		double v_[3];
} ;

using P = V ; // point
using C = V ; // color

inline V operator + ( const V& u, const V& v ) { return V( u.x()+v.x(), u.y()+v.y(), u.z()+v.z() ) ; }
inline V operator - ( const V& u, const V& v ) { return V( u.x()-v.x(), u.y()-v.y(), u.z()-v.z() ) ; }
inline V operator * ( const V& u, const V& v ) { return V( u.x()*v.x(), u.y()*v.y(), u.z()*v.z() ) ; }
inline V operator * ( double t, const V& v )   { return V( t*v.x(), t*v.y(), t*v.z() ) ; }
inline V operator / ( V v, double t )          { return ( 1/t )*v ; }

inline double dot( const V& u, const V& v )    { return u.x()*v.x()+u.y()*v.y()+u.z()*v.z() ; }
inline V cross( const V& u, const V& v )       { return V( u.y()*v.z()-u.z()*v.y(), u.z()*v.x()-u.x()*v.z(), u.x()*v.y()-u.y()*v.x() ) ; }
inline V unitV( V v )                          { return v/v.len() ; }

// random V in unit sphere
V rndVin1sphere()                  { while ( true ) { auto v = V::rnd( -1, 1 ) ; if ( 1>v.dot() ) return v ; } }
// random V on unit sphere (chapter 8.5)
V rndVon1sphere()                  { return unitV( rndVin1sphere() ) ; }
// random V against ray (chapter 8.6)
V rndVoppraydir( const V& normal ) { auto v = rndVin1sphere() ; return dot( v, normal ) ? v : -v ; }
// random V in unit disk (chapter 12.2)
V rndVin1disk() { while ( true ) { auto v = V( rnd( -1, 1 ), rnd( -1, 1 ), 0 ) ; if ( 1>v.dot() ) continue ; return v ; } }

V reflect( const V& v, const V& n )              { return v-2*dot( v, n )*n ; }
V refract( const V& v, const V& n, double qeta ) { auto tta = fmin( dot( -v, n ), 1. ) ; V perp = qeta*( v+tta*n ) ; V parl = -sqrt( fabs( 1.-perp.dot() ) )*n ; return perp+parl ; }

inline std::ostream& operator << ( std::ostream &out, const V &v ) { return out << v.x() << ' ' << v.y() << ' ' << v.z() ; }

#endif // V_H
