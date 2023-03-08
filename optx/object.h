#ifndef OBJECT_H
#define OBJECT_H

// system includes
#include <tuple>
#include <string>
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
		const Mesh operator[] ( unsigned int m ) ;

		Object( const std::string& wavefront ) ;

		size_t size() const { return ices_.size() ; } ;

	private:
		std::vector<float3>             vces_ ; // vertices shared by shapes
		std::vector<std::vector<uint3>> ices_ ; // indexed triangles (shapes)

		void procWavefrontObj( const std::string& wavefront ) ;
} ;

#endif // OBJECT_H
