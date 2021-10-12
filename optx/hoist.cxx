// system includes
#include <vector>

// subsystem includes
// CUDA
#include <vector_functions.h>
#include <vector_types.h>

// local includes
#include "thing.h"
#include "util.h"

// file specific includes
#include "hoist.h"

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

void Hoist::copyVcesToDevice( const std::vector<float3>& vertices ) {
	const size_t vertices_size = sizeof( float3 )*static_cast<unsigned int>( vertices.size() ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &vces ), vertices_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
		reinterpret_cast<void*>( vces ),
		vertices.data(),
		vertices_size,
		cudaMemcpyHostToDevice
		) ) ;
}

void Hoist::copyIcesToDevice( const std::vector<uint3>& indices ) {
	const size_t indices_size = sizeof( uint3 )*static_cast<unsigned int>( indices.size() ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ices ), indices_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
		reinterpret_cast<void*>( ices ),
		indices.data(),
		indices_size,
		cudaMemcpyHostToDevice
		) ) ;
}
