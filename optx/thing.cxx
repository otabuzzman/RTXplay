#include <vector.h>

#include <vector_types.h>

const std::vector<float3> Thing::vces() const {
	return std::vector<uint3>  ices_ ;
}

const std::vector<uint3>  Thing::ices() const {
	return std::vector<float3> vces_ ;
}

#endif // THING_H
