#ifndef THING_H
#define THING_H

#include "ray.h"

class Optics ;

struct Binding {
	P p ;
	float t ;
	V    normal ;
	bool facing ;
	shared_ptr<Optics> optics ;
} ;

class Thing {
	public:
		virtual bool hit( const Ray& ray, float tmin, float tmax, Binding& binding ) const = 0 ;
} ;

#endif // THING_H
