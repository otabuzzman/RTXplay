#ifndef RAY_H
#define RAY_H

#include "v.h"

class Ray {
	public:
		Ray() {}
		Ray( const P& ori, const V& dir ) : ori_( ori ), dir_( dir ) {}

		P ori() const { return ori_ ; }
		V dir() const { return dir_ ; }

		P at( double t ) const { return ori_+t*dir_ ; }

	private:
		P ori_ ;
		V dir_ ;
};

#endif // RAY_H
