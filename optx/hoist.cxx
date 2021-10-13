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
	num_vces = static_cast<unsigned int>( vertices.size() ) ;
	num_ices = static_cast<unsigned int>( indices.size()  ) ;

	makeGas( vertices, indices ) ;
}

Hoist::~Hoist() noexcept ( false ) {
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( vces ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( ices ) ) ) ;

	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( as_outbuf_ ) ) ) ;
}

void Hoist::makeGas( const std::vector<float3>& vertices, const std::vector<uint3>& indices ) {
	copyVcesToDevice( vertices ) ;
	copyIcesToDevice( indices  ) ;

	OptixBuildInput obi_thing = {} ;
	obi_thing.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES ;

	obi_thing.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3 ;
	obi_thing.triangleArray.numVertices                 = num_vces ;
	obi_thing.triangleArray.vertexBuffers               = reinterpret_cast<CUdeviceptr*>( &vces ) ;

	obi_thing.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ;
	obi_thing.triangleArray.numIndexTriplets            = num_ices ;
	obi_thing.triangleArray.indexBuffer                 = reinterpret_cast<CUdeviceptr>( ices ) ;

	const unsigned int obi_thing_flags[1]               = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT } ;
	obi_thing.triangleArray.flags                       = &obi_thing_flags[0] ;

	obi_thing.triangleArray.numSbtRecords               = 1 ; // number of SBT records in Hit Group section
	obi_thing.triangleArray.sbtIndexOffsetBuffer        = 0 ;
	obi_thing.triangleArray.sbtIndexOffsetSizeInBytes   = 0 ;
	obi_thing.triangleArray.sbtIndexOffsetStrideInBytes = 0 ;

	OptixAccelBuildOptions oas_options                  = {} ;
	oas_options.buildFlags                              = OPTIX_BUILD_FLAG_NONE ;
	oas_options.operation                               = OPTIX_BUILD_OPERATION_BUILD ;

	OptixAccelBufferSizes as_buffer_sizes ;
	OPTX_CHECK( optixAccelComputeMemoryUsage(
				optx_context,
				&oas_options,
				&obi_thing,
				1,
				&as_buffer_sizes
				) ) ;

	CUdeviceptr as_tmpbuf ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &as_tmpbuf  ), as_buffer_sizes.tempSizeInBytes ) ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &as_outbuf_ ), as_buffer_sizes.outputSizeInBytes ) ) ;

	OPTX_CHECK( optixAccelBuild(
				optx_context,
				0,
				&oas_options,
				&obi_thing,
				1,
				as_tmpbuf,
				as_buffer_sizes.tempSizeInBytes,
				as_outbuf_,
				as_buffer_sizes.outputSizeInBytes,
				&as_handle,
				nullptr, 0
				) ) ;

	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( as_tmpbuf ) ) ) ;
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
