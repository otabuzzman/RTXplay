#ifndef SCENE_H
#define SCENE_H

// system includes
#include <memory>
#include <vector>

// subsystem includes
// none

// local includes
#include "hoist.h"

// file specific includes
// none

class Scene {
	public:
		std::shared_ptr<Hoist> operator[] ( unsigned int i ) { return things_[i] ; } ;

		void   load() ;
		size_t size() const ;

	private:
		std::vector<std::shared_ptr<Hoist>> things_ ;
} ;

#endif // SCENE_H
