#ifndef THINGS_H
#define THINGS_H

#include "Thing.h"

#include <memory>
#include <vector>

using std::shared_ptr ;
using std::make_shared ;

class Things : public Thing {
	public:
		Things() {}
		Things( shared_ptr<Thing> thing ) { add( thing ) ; }

		void clear() { things.clear() ; }
		void add( shared_ptr<Thing> thing ) { things.push_back( thing ) ; }

		virtual bool hit( const Ray& ray, double tmin, double tmax, record& rec ) const override ;

	private:
		std::vector<shared_ptr<Thing>> things ;
} ;

bool Things::hit( const Ray& ray, double tmin, double tmax, record& rec ) const {
	record nrec ;
	bool shot = false ;
	auto tact = tmax ;

	for ( const auto& thing : things ) {
		if ( thing->hit( ray, tmin, tact, nrec ) ) {
			shot = true ;
			tact = nrec.t ;
			rec  = nrec ;
		}
	}

	return shot ;
}

#endif