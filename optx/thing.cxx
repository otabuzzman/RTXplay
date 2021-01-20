#include <vector>

#include <vector_types.h>

#include "thing.h"

const std::vector<float3> Thing::vces() const {
	return vces_ ;
}

const std::vector<uint3>  Thing::ices() const {
	return ices_ ;
}
