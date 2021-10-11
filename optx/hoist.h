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
	virtual ~Hoist() noexcept ( false ) ;

	void copyVcesToDevice( const std::vector<float3>& vces ) ;
	void copyIcesToDevice( const std::vector<uint3>&  ices ) ;

	unsigned int num_vces = 0 ;
	unsigned int num_ices = 0 ;
} ;

#endif // HOIST_H
