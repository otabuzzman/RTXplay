#ifndef THINGS_H
#define THINGS_H

#include <memory>
#include <vector>

#include "ray.h"
#include "thing.h"

class Things : public Thing {
	public:
		Things() {}
		Things( std::shared_ptr<Thing> thing ) { add( thing ) ; }

		void clear() { things_.clear() ; }
		void add( std::shared_ptr<Thing> thing ) { things_.push_back( thing ) ; }

		virtual bool hit( const Ray& ray, double tmin, double tmax, Binding& binding ) const override ;

	private:
		std::vector<std::shared_ptr<Thing>> things_ ;
} ;

bool Things::hit( const Ray& ray, double tmin, double tmax, Binding& binding ) const {
	Binding buffer ;
	bool shot = false ;
	auto tact = tmax ;

	for ( const auto& thing : things_ ) {
		if ( thing->hit( ray, tmin, tact, buffer ) ) {
			shot = true ;
			tact = buffer.t ;
			binding = buffer ;
		}
	}

	return shot ;
}

#endif // THINGS_H
