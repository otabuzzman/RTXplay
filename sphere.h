#ifndef SPHERE_H
#define SPHERE_H

#include "thing.h"

class Sphere : public Thing {
	public:
		Sphere() {}
		Sphere( const P center, double radius, shared_ptr<Optics> optics ) : center_( center ), radius_( radius ), optics_( optics ) {}

		P      center() const { return center_ ; }
		double radius() const { return radius_ ; }

		virtual bool hit( const Ray& ray, double tmin, double tmax, Bucket& bucket ) const override ;

	private:
		P center_ ;
		double radius_ ;
		shared_ptr<Optics> optics_ ;
} ;

bool Sphere::hit( const Ray& ray, double tmin, double tmax, Bucket& bucket ) const {
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

	bucket.t = t ;
	bucket.p = ray.at( bucket.t ) ;
	V outward = ( bucket.p-center_ )/radius_ ;
	bucket.facing = 0>dot( ray.dir(), outward ) ;
	bucket.normal = bucket.facing ? outward : -outward ;
	bucket.optics = optics_ ;

	return true ;
}

#endif
