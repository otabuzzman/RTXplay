#ifndef HOIST_H
#define HOIST_H

// system includes
#include <vector>

// subsystem includes
// CUDA
#include <vector_types.h>

// local includes
#include "thing.h"

// file specific includes
// none

struct Hoist : public Thing {
	Hoist( const std::vector<float3>& vertices, const std::vector<uint3>& indices ) ;
	~Hoist() noexcept ( false ) ;

	unsigned int num_vces = 0 ;
	unsigned int num_ices = 0 ;

	private:
		void copyVcesToDevice( const std::vector<float3>& vertices ) ;
		void copyIcesToDevice( const std::vector<uint3>&  indices  ) ;
} ;

#endif // HOIST_H
