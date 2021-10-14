#ifndef HOIST_H
#define HOIST_H

// system includes
#include <vector>

// subsystem includes
// OptiX
#include <optix.h>
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

	OptixTraversableHandle as_handle = 0 ;

	private:
		void makeGas() ;

		void copyVcesToDevice( const std::vector<float3>& vertices ) ;
		void copyIcesToDevice( const std::vector<uint3>&  indices  ) ;

		CUdeviceptr as_outbuf_ = 0 ;
} ;

#endif // HOIST_H
