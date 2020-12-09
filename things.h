#ifndef THINGS_H
#define THINGS_H

#include "thing.h"

#include <memory>
#include <vector>

using std::shared_ptr ;
using std::make_shared ;

class Things : public Thing {
	public:
		Things() {}
		Things( shared_ptr<Thing> thing ) { add( thing ) ; }

		void clear() { things_.clear() ; }
		void add( shared_ptr<Thing> thing ) { things_.push_back( thing ) ; }

		virtual bool hit( const Ray& ray, double tmin, double tmax, Bucket& bucket ) const override ;

	private:
		std::vector<shared_ptr<Thing>> things_ ;
} ;

bool Things::hit( const Ray& ray, double tmin, double tmax, Bucket& bucket ) const {
	Bucket buffer ;
	bool shot = false ;
	auto tact = tmax ;

	for ( const auto& thing : things_ ) {
		if ( thing->hit( ray, tmin, tact, buffer ) ) {
			shot = true ;
			tact = buffer.t ;
			bucket  = buffer ;
		}
	}

	return shot ;
}

#endif