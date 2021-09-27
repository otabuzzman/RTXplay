#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "thing.h"

class Scene {
	public:
		Thing operator[] ( unsigned int i ) { return things_[i] ; } ;

		void   load() ;
		size_t size() const ;

	private:
		std::vector<Thing> things_ ;
} ;

#endif // SCENE_H
