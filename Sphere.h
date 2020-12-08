#ifndef SPHERE_H
#define SPHERE_H

#include "Thing.h"

class Sphere : public Thing {
	public:
		Sphere() {}
		Sphere( P center, double radius, shared_ptr<Optics> optics ) : center_( center ), radius_( radius ), optics_( optics ) {}

		P      center() { return center_ ; }
		double radius() const { return radius_ ; }

		virtual bool hit( const Ray& ray, double tmin, double tmax, Payload& payload ) const override ;

	private:
		P center_ ;
		double radius_ ;
		shared_ptr<Optics> optics_ ;
} ;

bool Sphere::hit( const Ray& ray, double tmin, double tmax, Payload& payload ) const {
	V o = ray.ori()-center_ ;
	auto a = ray.dir().len2() ;    // simplified quadratic equation (see also sphere() in main.cpp)
	auto b = dot( ray.dir(), o ) ;
	auto c = o.len2()-radius_*radius_ ;
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

	payload.t = t ;
	payload.p = ray.at( payload.t ) ;
	V outward = ( payload.p-center_ )/radius_ ;
	payload.facing = 0>dot( ray.dir(), outward ) ;
	payload.normal = payload.facing ? outward : -outward ;
	payload.optics = optics_ ;

	return true ;
}

#endif