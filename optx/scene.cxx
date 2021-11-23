// system includes
#include <cstring>
#include <tuple>

// subsystem includes
// CUDA
#include <vector_functions.h>
#include <vector_types.h>

// local includes
// none

// file specific includes
#include "scene.h"

template<typename T>
void copyVIToDevice( CUdeviceptr& to, const T* from, unsigned int num ) {
	const size_t to_size = sizeof( T )*num ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &to ), to_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
		reinterpret_cast<void*>( to ),
		from,
		to_size,
		cudaMemcpyHostToDevice
		) ) ;
}

Scene::Scene( const OptixDeviceContext& optx_context ) : optx_context_( optx_context ), is_outbuf_( 0 ), d_ises_( 0 ) {
}

Scene::~Scene() noexcept ( false ) {
	for ( auto b : as_outbuf_ ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;
	for ( auto b : vces_ ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;
	for ( auto b : ices_ ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;
	if ( is_outbuf_ ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( is_outbuf_ ) ) ) ;
	if ( d_ises_ )    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_ises_ ) ) ) ;
}

unsigned int Scene::add( Object& object ) {
	const unsigned int id = static_cast<unsigned int>( as_handle_.size() ) ;

	// collect object's submeshes into one
	std::vector<float3> o_vces ;
	std::vector<uint3>  o_ices ;

	// submesh tuple
	float3*      vces ;
	unsigned int vces_size ;
	uint3*       ices ;
	unsigned int ices_size ;

	for ( unsigned int o = 0 ; object.size()>o ; o++ ) {
		std::tie( vces, vces_size, ices, ices_size ) = object[o] ;
		for ( unsigned int v = 0 ; vces_size>v ; v++ )
			o_vces.push_back( vces[v] ) ;
		for ( unsigned int i = 0 ; ices_size>i ; i++ )
			o_ices.push_back( ices[i] ) ;
	}

	// setup this object's build input structure
	OptixBuildInput obi_object = {} ;
	obi_object.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES ;

	CUdeviceptr d_vces = 0 ;
	const unsigned int o_vces_size = static_cast<unsigned int>( o_vces.size() ) ;
	copyVIToDevice<float3>( d_vces, o_vces.data(), o_vces_size ) ;
	vces_.push_back( d_vces ) ;
	obi_object.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3 ;
	obi_object.triangleArray.numVertices                 = o_vces_size ;
	obi_object.triangleArray.vertexBuffers               = &vces_[id] ;

	CUdeviceptr d_ices = 0 ;
	const unsigned int o_ices_size = static_cast<unsigned int>( o_ices.size() ) ;
	copyVIToDevice<uint3>( d_ices, o_ices.data(), o_ices_size ) ;
	ices_.push_back( d_ices ) ;
	obi_object.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ;
	obi_object.triangleArray.numIndexTriplets            = o_ices_size ;
	obi_object.triangleArray.indexBuffer                 = ices_[id] ;

	const unsigned int obi_object_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT } ;
	obi_object.triangleArray.flags                       = &obi_object_flags[0] ;
	obi_object.triangleArray.numSbtRecords               = 1 ; // number of SBT records in Hit Group section
	obi_object.triangleArray.sbtIndexOffsetBuffer        = 0 ;
	obi_object.triangleArray.sbtIndexOffsetSizeInBytes   = 0 ;
	obi_object.triangleArray.sbtIndexOffsetStrideInBytes = 0 ;

	OptixAccelBuildOptions oas_options = {} ;
	oas_options.buildFlags             = OPTIX_BUILD_FLAG_NONE ;
	oas_options.operation              = OPTIX_BUILD_OPERATION_BUILD ;

	// request acceleration structure buffer sizes from OptiX
	OptixAccelBufferSizes as_buffer_sizes ;
	OPTX_CHECK( optixAccelComputeMemoryUsage(
				optx_context_,
				&oas_options,
				&obi_object,
				1,
				&as_buffer_sizes
				) ) ;

	// allocate GPU memory for acceleration structure buffers
	CUdeviceptr as_tmpbuf, as_outbuf ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &as_tmpbuf ), as_buffer_sizes.tempSizeInBytes ) ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &as_outbuf ), as_buffer_sizes.outputSizeInBytes ) ) ;

	// allocate GPU memory for acceleration structure compaction buffer
//	CUdeviceptr as_zipbuf ;
//	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &as_zipbuf ), as_buffer_sizes.outputSizeInBytes ) ) ;
	// allocate GPU memory for acceleration structure compaction buffer size
//	CUdeviceptr as_zipbuf_size ;
//	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &as_zipbuf_size ), sizeof( unsigned long long ) ) ) ;

	// provide request description for acceleration structure compaction buffer size
//	OptixAccelEmitDesc oas_request ;
//	oas_request.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE ;
//	oas_request.result = as_zipbuf_size ;

	// build acceleration structure
	OptixTraversableHandle as_handle ;
	OPTX_CHECK( optixAccelBuild(
				optx_context_,
				0,
				&oas_options,
				&obi_object,
				1,
				as_tmpbuf,
				as_buffer_sizes.tempSizeInBytes,
				as_outbuf,
				// acceleration structure compaction (must comment previous line)
//				as_zipbuf,
				as_buffer_sizes.outputSizeInBytes,
				&as_handle,
				nullptr, 0
				// acceleration structure compaction (must comment previous line)
//				&oas_request, 1
				) ) ;
	as_handle_.push_back( as_handle ) ;

	// retrieve acceleration structure compaction buffer size from GPU memory
//	unsigned long long as_zipbuf_size ;
//	CUDA_CHECK( cudaMemcpy(
//				reinterpret_cast<void*>( &as_zipbuf_size ),
//				reinterpret_cast<void*>( as_zipbuf_size ),
//				sizeof( unsigned long long ),
//				cudaMemcpyDeviceToHost
//				) ) ;

	// condense previously built acceleration structure
//	OPTX_CHECK( optixAccelCompact(
//				optx_context_,
//				0,
//				as_handle,
//				as_outbuf,
//				as_zipbuf_size,
//				&as_handle ) ) ;

	as_outbuf_.push_back( as_outbuf ) ;

	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( as_tmpbuf ) ) ) ;
	// free GPU memory for acceleration structure compaction buffer
//	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( as_zipbuf      ) ) ) ;
//	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( as_zipbuf_size ) ) ) ;

	return id ;
}

unsigned int Scene::add( Thing& thing, const float* transform, unsigned int object ) {
	const unsigned int id = static_cast<unsigned int>( things_.size() ) ;

	OptixInstance instance     = {} ;

	instance.instanceId        = id ;
	instance.visibilityMask    = 255 ; // OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK

	instance.flags             = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;
	instance.sbtOffset         = id ;

	const size_t transform_size = sizeof( float )*12 ;
	memcpy( instance.transform, transform, transform_size ) ;

	instance.traversableHandle = as_handle_[object] ;
	h_ises_.push_back( instance ) ;

	thing.vces = reinterpret_cast<float3*>( vces_[object] ) ;
	thing.ices = reinterpret_cast<uint3*>( ices_[object] ) ;
	things_.push_back( thing ) ;

	return id ;
}

void Scene::build( OptixTraversableHandle* handle ) {
	const size_t instances_size = sizeof( OptixInstance )*h_ises_.size() ;
	CUDA_CHECK( cudaMalloc( (void**) &d_ises_, instances_size) );
	CUDA_CHECK( cudaMemcpy( (void*) d_ises_, h_ises_.data(), instances_size, cudaMemcpyHostToDevice ) );

	// create build input structure for thing instances
	OptixBuildInput obi_things            = {} ;
	obi_things.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	obi_things.instanceArray.instances    = d_ises_ ;
	obi_things.instanceArray.numInstances = static_cast<unsigned int>( h_ises_.size() ) ;

	OptixAccelBuildOptions ois_options    = {} ;
	ois_options.buildFlags                = OPTIX_BUILD_FLAG_NONE ;
	ois_options.operation                 = OPTIX_BUILD_OPERATION_BUILD ;

	// request acceleration structure buffer sizes from OptiX
	OptixAccelBufferSizes is_buffer_sizes ;
	OPTX_CHECK( optixAccelComputeMemoryUsage(
				optx_context_,
				&ois_options,
				&obi_things,
				1,
				&is_buffer_sizes
				) ) ;

	// allocate GPU memory for acceleration structure buffers
	CUdeviceptr is_tmpbuf ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &is_tmpbuf ), is_buffer_sizes.tempSizeInBytes ) ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &is_outbuf_ ), is_buffer_sizes.outputSizeInBytes ) ) ;

	OPTX_CHECK( optixAccelBuild(
				optx_context_,
				0,
				&ois_options,
				&obi_things,
				1,
				is_tmpbuf,
				is_buffer_sizes.tempSizeInBytes,
				is_outbuf_,
				is_buffer_sizes.outputSizeInBytes,
				handle,
				nullptr, 0
				) ) ;

	CUDA_CHECK( cudaFree( (void*) is_tmpbuf ) ) ;
}
