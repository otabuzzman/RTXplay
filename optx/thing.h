#ifndef THING_H
#define THING_H

#include <vector>

#include <vector_functions.h>
#include <vector_types.h>

#include "optics.h"

class Thing {
	public:
		const std::vector<float3> vces() const {
			return vces_ ;
		}

		const std::vector<uint3>  ices() const {
			return ices_ ;
		}

		const Optics optics() const {
			return optics_ ;
		}

	protected:
		std::vector<float3> vces_ ; // unique vertices
		std::vector<uint3>  ices_ ; // indexed triangles
		Optics optics_ ;            // optical properties
} ;

#endif // THING_H
