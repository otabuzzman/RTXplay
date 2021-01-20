#ifndef THING_H
#define THING_H

#include "ray.h"

class Optics ;

struct Binding {
	P p ;
	double t ;
	V    normal ;
	bool facing ;
	shared_ptr<Optics> optics ;
} ;

class Thing {
	public:
		virtual bool hit( const Ray& ray, const double tmin, const double tmax, Binding& binding ) const = 0 ;
} ;

#endif // THING_H
