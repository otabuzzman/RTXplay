#ifndef SPHERE_H
#define SPHERE_H

#include "Thing.h"
#include "V.h"

class Sphere : public Thing {
	public:
		Sphere() {}
		Sphere( P center, double radius ) : cen(center), rad(radius) {}

		P      center() { return cen ; }
		double radius() const { return rad ; }

		virtual bool hit( const Ray& ray, double tmin, double tmax, record& rec ) const override ;

	private:
		P cen ;
		double rad ;
} ;

bool Sphere::hit( const Ray& ray, double tmin, double tmax, record& rec ) const {
	V o = ray.ori()-cen ;
	auto a = ray.dir().plen() ;    // simplified quadratic equation (see also sphere() in main.cpp)
	auto b = dot( ray.dir(), o ) ;
	auto c = o.plen()-rad*rad ;
	auto discriminant = b*b-a*c ;

	if ( 0>discriminant )
		return false ;
	// else
	auto x = sqrt( discriminant ) ;

	// nearest t in range
	auto t = ( -b-x )/a ;
	if ( tmin>t || t>tmax ) {
		t = ( -b+x )/a ;
		if ( tmin>t || t>tmax )
			return false ;
	}

	rec.t = t ;
	rec.p = ray.at( rec.t ) ;
	V outward = ( rec.p-cen )/rad ;
	rec.setnormal( ray, outward ) ;

	return true ;
}

#endif