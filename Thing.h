#ifndef THING_H
#define THING_H

#include "Ray.h"

class Optics ;

struct Payload {
	P p ;
	double t ;
	V    normal ;
	bool facing ;
	shared_ptr<Optics> optics ;
} ;

class Thing {
	public:
		virtual bool hit( const Ray& ray, double tmin, double tmax, Payload& payload ) const = 0 ;
} ;

#endif