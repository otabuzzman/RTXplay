#ifndef THING_H
#define THING_H

#include "Ray.h"

struct record {
	P p ;
	V normal ;
	double t ;
	bool facing ;

	inline void setnormal( const Ray& ray, const V& outward ) {
		facing = 0>dot( ray.dir(), outward ) ;
		normal = facing ? outward : -outward ;
	}
} ;

class Thing {
	public:
		virtual bool hit( const Ray& ray, double tmin, double tmax, record& rec ) const = 0 ;
} ;

#endif