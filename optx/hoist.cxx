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

Hoist::~Hoist() noexcept ( false ) {
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( vces ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( ices ) ) ) ;
}

void Hoist::copyVcesToDevice( const std::vector<float3>& data ) {
	const size_t data_size = sizeof( float3 )*static_cast<unsigned int>( data.size() ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &vces ), data_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
		reinterpret_cast<void*>( vces ),
		data.data(),
		data_size,
		cudaMemcpyHostToDevice
		) ) ;
}

void Hoist::copyIcesToDevice( const std::vector<uint3>& data ) {
	const size_t data_size = sizeof( uint3 )*static_cast<unsigned int>( data.size() ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ices ), data_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
		reinterpret_cast<void*>( ices ),
		data.data(),
		data_size,
		cudaMemcpyHostToDevice
		) ) ;
}
