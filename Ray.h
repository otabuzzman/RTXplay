#ifndef RAY_H
#define RAY_H

#include "V.h"

class Ray {
	public:
		Ray() {}
		Ray( const P& ori, const V& dir ) : o( ori ), d( dir ) {}

		P ori() const { return o ; }
		V dir() const { return d ; }

		P at( double t ) const { return o+t*d ; }

	private:
		P o ;
		V d ;
};

#endif