#ifndef SPHERE_H
#define SPHERE_H

#include "thing.h"

class Sphere : public Thing {
	public:
		Sphere() {}
		Sphere( const P& center, const double radius, shared_ptr<Optics> optics ) : center_( center ), radius_( radius ), optics_( optics ) {}

		virtual bool hit( const Ray& ray, const double tmin, const double tmax, Binding& binding ) const override ;

	private:
		P center_ ;
		double radius_ ;
		shared_ptr<Optics> optics_ ;
} ;

bool Sphere::hit( const Ray& ray, const double tmin, const double tmax, Binding& binding ) const {
	V o = ray.ori()-center_ ;
	auto a = ray.dir().dot() ;    // simplified quadratic equation (see also sphere() in main.cpp)
	auto b = dot( ray.dir(), o ) ;
	auto c = o.dot()-radius_*radius_ ;
	auto discriminant = b*b-a*c ;

	if ( 0>discriminant )
		return false ;

	auto x = sqrt( discriminant ) ;

	// nearest t in range
	auto t = ( -b-x )/a ;
	if ( tmin>t || t>tmax ) {
		t = ( -b+x )/a ;
		if ( tmin>t || t>tmax )
			return false ;
	}

	binding.t = t ;
	binding.p = ray.at( binding.t ) ;
	V outward = ( binding.p-center_ )/radius_ ;
	binding.facing = 0>dot( ray.dir(), outward ) ;
	binding.normal = binding.facing ? outward : -outward ;
	binding.optics = optics_ ;

	return true ;
}

#endif // SPHERE_H
