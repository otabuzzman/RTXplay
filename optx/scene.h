#ifndef SCENE_H
#define SCENE_H

// system includes
#include <memory>
#include <vector>

// subsystem includes
// none

// local includes
#include "hoist.h"
#include "sphere.h"

// file specific includes
// none

class Scene {
	public:
		const Hoist& operator[] ( unsigned int i ) { return things_[i] ; } ;

		void         load() ;
		const Hoist* data() const { return things_.data() ; } ;
		size_t       size() const { return things_.size() ; } ;

	private:
		std::vector<std::shared_ptr<Sphere>> meshes_ ; // unique triangle meshes (UTM)
		std::vector<Hoist>                   things_ ;
} ;

#endif // SCENE_H
