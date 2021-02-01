#ifndef THING_H
#define THING_H

#include <memory>

#include "optics.h"
#include "ray.h"
#include "v.h"

class Optics ;

struct Binding {
	P p ;
	double t ;
	V    normal ;
	bool facing ;
	std::shared_ptr<Optics> optics ;
} ;

class Thing {
	public:
		virtual bool hit( const Ray& ray, const double tmin, const double tmax, Binding& binding ) const = 0 ;
} ;

#endif // THING_H
