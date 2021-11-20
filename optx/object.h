#ifndef OBJECT_H
#define OBJECT_H

// system includes
#include <tuple>
#include <vector>

// subsystem includes
// CUDA
#include <vector_types.h>

// local includes
// none

// file specific includes
// none

typedef std::tuple<float3*, unsigned int, uint3*, unsigned int> Mesh ;

class Object {
	public:
		const Mesh operator[] ( unsigned int i ) ;

		Object( const std::string& wavefront ) ;

		const size_t size() { return vces_.size() ; } ;

	private:
		std::vector<std::vector<float3>> vces_ ; // submesh's unique vertices...
		std::vector<std::vector<uint3>>  ices_ ; // ...as indexed triangles

		void procWavefrontObj( const std::string& wavefront ) ;
} ;

#endif // OBJECT_H
