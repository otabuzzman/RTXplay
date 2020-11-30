#ifndef V_H
#define V_H

#include <cmath>
#include <iostream>

using std::sqrt ;

class V {
	public:
		V() : m{ 0, 0, 0 } {}
		V( double x, double y, double z ) : m{ x, y, z } {}

		double x() const { return m[0] ; }
		double y() const { return m[1] ; }
		double z() const { return m[2] ; }

		V operator - () const                  { return V( -m[0], -m[1], -m[2] ) ; }

		double operator [] ( int i ) const     { return m[i] ; }
		double& operator [] ( int i )          { return m[i] ; }

		V& operator += ( const V &v )          { m[0] += v.m[0] ; m[1] += v.m[1] ; m[2] += v.m[2] ; return *this ; }
		V& operator *= ( const double t )      { m[0] *= t ; m[1] *= t ; m[2] *= t ; return *this ; }
		V& operator /= ( const double t )      { return *this *= 1/t ; }

		double len() const                     { return sqrt( m[0]*m[0]+m[1]*m[1]+m[2]*m[2] ) ; }

	private:
		double m[3]; 
} ;

using P = V ; // point
using C = V ; // color

inline V operator + (const V &u, const V &v)   { return V( u.x()+v.x(), u.y()+v.y(), u.z()+v.z() ) ; }
inline V operator - ( const V &u, const V &v ) { return V( u.x()-v.x(), u.y()-v.y(), u.z()-v.z() ) ; }
inline V operator * ( const V &u, const V &v ) { return V( u.x()*v.x(), u.y()*v.y(), u.z()*v.z() ) ; }
inline V operator * ( double t, const V &v )   { return V( t*v.x(), t*v.y(), t*v.z() ) ; }
inline V operator / ( V v, double t )          { return ( 1/t )*v ; }

inline double dot( const V &u, const V &v )    { return u.x()*v.x()+u.y()*v.y()+u.z()*v.z() ; }
inline V cross( const V &u, const V &v )       { return V( u.y()*v.z()-u.z()*v.y(), u.z()*v.x()-u.x()*v.z(), u.x()*v.y()-u.y()*v.x() ) ; }
inline V unitV( V v )                          { return v/v.len() ; }

inline std::ostream& operator << ( std::ostream &out, const V &v ) { return out << v.x() << ' ' << v.y() << ' ' << v.z() ; }

#endif
