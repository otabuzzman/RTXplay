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

		void clear() { things_.clear() ; }
		void add( shared_ptr<Thing> thing ) { things_.push_back( thing ) ; }

		virtual bool hit( const Ray& ray, double tmin, double tmax, Payload& payload ) const override ;

	private:
		std::vector<shared_ptr<Thing>> things_ ;
} ;

bool Things::hit( const Ray& ray, double tmin, double tmax, Payload& payload ) const {
	Payload tmp_payload ;
	bool shot = false ;
	auto tact = tmax ;

	for ( const auto& thing : things_ ) {
		if ( thing->hit( ray, tmin, tact, tmp_payload ) ) {
			shot = true ;
			tact = tmp_payload.t ;
			payload  = tmp_payload ;
		}
	}

	return shot ;
}

#endif