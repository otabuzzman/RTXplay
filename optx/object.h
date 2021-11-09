#ifndef OBJECT_H
#define OBJECT_H

// system includes
#include <vector>

// subsystem includes
// CUDA
#include <vector_types.h>

// local includes
#include "hoist.h"

// file specific includes
// none

class Object : public Hoist {
	public:
		Object( const std::string& wavefront ) ;
		~Object() noexcept ( false ) ;

		const std::vector<float3>& vces() const { return vces_ ; } ;
		const std::vector<uint3>&  ices() const { return ices_ ; } ;

	private:
		void procWavefrontObj( const std::string& wavefront ) ;

		void copyVcesToDevice() ;
		void copyIcesToDevice() ;

		std::vector<float3> vces_ ; // object's unique vertices...
		std::vector<uint3>  ices_ ; // ...as indexed triangles

		static int utm_count_ ;
} ;

#endif // OBJECT_H
