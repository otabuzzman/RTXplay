// system includes
#include <vector>

// subsystem includes
// OptiX
#include <optix.h>
#include <optix_stubs.h>
// CUDA
#include <vector_functions.h>
#include <vector_types.h>

// local includes
#include "util.h"

// file specific includes
#include "hoist.h"

// common globals
extern OptixDeviceContext optx_context ;

Hoist::Hoist( const std::vector<float3>& vertices, const std::vector<uint3>& indices ) {
	copyVcesToDevice( vertices ) ;
	num_vces = static_cast<unsigned int>( vertices.size() ) ;
	copyIcesToDevice( indices )  ;
	num_ices = static_cast<unsigned int>( indices.size()  ) ;
}

Hoist::~Hoist() noexcept ( false ) {
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( vces ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( ices ) ) ) ;
}
