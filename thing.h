#ifndef THING_H
#define THING_H

#include <memory>

#include "ray.h"

class Optics ;

struct Binding {
	double t ;
	P p ;
	V    normal ;
	bool facing ;
	std::shared_ptr<Optics> optics ;
} ;

class Thing {
	public:
		virtual bool hit( const Ray& ray, const double tmin, const double tmax, Binding& binding ) const = 0 ;

	protected:
		P center_ ;
		std::shared_ptr<Optics> optics_ ;
} ;

#endif // THING_H
