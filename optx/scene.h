#ifndef SCENE_H
#define SCENE_H

#include <memory>
#include <vector>

#include "thing.h"

class Scene {
	public:
		std::shared_ptr<Thing> operator[] ( unsigned int i ) { return things_[i] ; } ;

		void   load() ;
		size_t size() const ;

	private:
		std::vector<std::shared_ptr<Thing>> things_ ;
} ;

#endif // SCENE_H