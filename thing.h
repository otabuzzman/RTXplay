#ifndef THING_H
#define THING_H

#include "ray.h"

class Optics ;

struct Bucket {
	P p ;
	double t ;
	V    normal ;
	bool facing ;
	shared_ptr<Optics> optics ;
} ;

class Thing {
	public:
		virtual bool hit( const Ray& ray, double tmin, double tmax, Bucket& bucket ) const = 0 ;
} ;

#endif