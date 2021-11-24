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
void copyDataToDevice( CUdeviceptr& dst, const T* src, unsigned int num ) {
	const unsigned int size = sizeof( T )*num ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &dst ), size ) ) ;
	CUDA_CHECK( cudaMemcpy(
		reinterpret_cast<void*>( dst ),
		src,
		size,
		cudaMemcpyHostToDevice
		) ) ;
}

Scene::Scene( const OptixDeviceContext& optx_context ) : optx_context_( optx_context ), is_outbuf_( 0 ), ises_( 0 ), obi_things_( {} ) {
}

Scene::~Scene() noexcept ( false ) {
	for ( auto b : as_outbuf_ ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;
	for ( auto b : vces_ ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;
	for ( auto b : ices_ ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;
	free() ;
}

unsigned int Scene::add( Object& object ) {
	const unsigned int id = static_cast<unsigned int>( as_handle_.size() ) ;

	// concatenated submeshes
	std::vector<float3> as_vces ;
	std::vector<uint3>  as_ices ;

	// submesh tuple (Mesh)
	{
		float3*      vces ;
		unsigned int vces_size ;
		uint3*       ices ;
		unsigned int ices_size ;

		for ( unsigned int o = 0 ; object.size()>o ; o++ ) {
			std::tie( vces, vces_size, ices, ices_size ) = object[o] ;
			for ( unsigned int v = 0 ; vces_size>v ; v++ )
				as_vces.push_back( vces[v] ) ;
			for ( unsigned int i = 0 ; ices_size>i ; i++ )
				as_ices.push_back( ices[i] ) ;
		}
	}

	// set up this object's build input structure
	OptixBuildInput obi_object = {} ;
	obi_object.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES ;

	CUdeviceptr vces = 0 ;
	const unsigned int as_vces_size = static_cast<unsigned int>( as_vces.size() ) ;
	copyDataToDevice<float3>( vces, as_vces.data(), as_vces_size ) ;
	vces_.push_back( vces ) ;
	obi_object.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3 ;
	obi_object.triangleArray.numVertices                 = as_vces_size ;
	obi_object.triangleArray.vertexBuffers               = &vces_[id] ;

	CUdeviceptr ices = 0 ;
	const unsigned int as_ices_size = static_cast<unsigned int>( as_ices.size() ) ;
	copyDataToDevice<uint3>( ices, as_ices.data(), as_ices_size ) ;
	ices_.push_back( ices ) ;
	obi_object.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ;
	obi_object.triangleArray.numIndexTriplets            = as_ices_size ;
	obi_object.triangleArray.indexBuffer                 = ices_[id] ;

	const unsigned int obi_object_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT } ;
	obi_object.triangleArray.flags                       = &obi_object_flags[0] ;
	obi_object.triangleArray.numSbtRecords               = 1 ; // number of SBT records in Hit Group section
	obi_object.triangleArray.sbtIndexOffsetBuffer        = 0 ;
	obi_object.triangleArray.sbtIndexOffsetSizeInBytes   = 0 ;
	obi_object.triangleArray.sbtIndexOffsetStrideInBytes = 0 ;

	// GAS options
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
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &as_tmpbuf ), as_buffer_sizes.tempSizeInBytes   ) ) ;
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
//	unsigned long long zipbuf_size ;
//	CUDA_CHECK( cudaMemcpy(
//				reinterpret_cast<void*>( &zipbuf_size ),
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
//				zipbuf_size,
//				&as_handle ) ) ;

	as_outbuf_.push_back( as_outbuf ) ;

	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( as_tmpbuf ) ) ) ;
	// free GPU memory for acceleration structure compaction buffer
//	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( as_zipbuf      ) ) ) ;
//	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( as_zipbuf_size ) ) ) ;

	return id ;
}

unsigned int Scene::add( Thing& thing, unsigned int object ) {
	const unsigned int id = static_cast<unsigned int>( things_.size() ) ;

	// set up this things's instance structure
	OptixInstance ois_thing     = {} ;
	ois_thing.instanceId        = id ;
	ois_thing.visibilityMask    = 255 ; // OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK

	ois_thing.flags             = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;
	ois_thing.sbtOffset         = id ;

	const float transform[12] = {
		1., 0., 0., 0.,
		0., 1., 0., 0.,
		0., 0., 1., 0.
	} ;
	memcpy( ois_thing.transform, transform, sizeof( float )*12 ) ;

	ois_thing.traversableHandle = as_handle_[object] ;

	is_ises_.push_back( ois_thing ) ;

	thing.vces = reinterpret_cast<float3*>( vces_[object] ) ;
	thing.ices = reinterpret_cast<uint3*> ( ices_[object] ) ;
	things_.push_back( thing ) ;

	return id ;
}

bool Scene::set( unsigned int thing, const float* transform, bool update ) {
	if ( is_ises_.size()>thing ) {
		OptixInstance* is_instance = &is_ises_[thing] ;
		memcpy( &is_instance->transform, transform, sizeof( float )*12 ) ;

		if ( update ) {
			OptixInstance* instance = &reinterpret_cast<OptixInstance*>( ises_ )[thing] ;
			CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( instance ),
				is_instance,
				sizeof( OptixInstance ),
				cudaMemcpyHostToDevice
				) ) ;
		}
		return true ;
	} else
		return false ;
}

void Scene::build( OptixTraversableHandle* is_handle ) {
	free() ;
	// set up build input structure for thing instances
	copyDataToDevice<OptixInstance>( ises_, is_ises_.data(), static_cast<unsigned int>( is_ises_.size() ) ) ;
	obi_things_.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	obi_things_.instanceArray.instances    = ises_ ;
	obi_things_.instanceArray.numInstances = static_cast<unsigned int>( is_ises_.size() ) ;

	// IAS options
	OptixAccelBuildOptions ois_options    = {} ;
	ois_options.buildFlags                = OPTIX_BUILD_FLAG_ALLOW_UPDATE ;
	ois_options.operation                 = OPTIX_BUILD_OPERATION_BUILD ;

	// request acceleration structure buffer sizes from OptiX
	OptixAccelBufferSizes is_buffer_sizes ;
	OPTX_CHECK( optixAccelComputeMemoryUsage(
				optx_context_,
				&ois_options,
				&obi_things_,
				1,
				&is_buffer_sizes
				) ) ;

	// allocate GPU memory for acceleration structure buffers
	CUdeviceptr is_tmpbuf ;
	is_updbuf_size_ = is_buffer_sizes.tempUpdateSizeInBytes ;
	is_outbuf_size_ = is_buffer_sizes.tempUpdateSizeInBytes ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &is_tmpbuf  ), is_buffer_sizes.tempSizeInBytes ) ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &is_updbuf_ ), is_updbuf_size_                 ) ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &is_outbuf_ ), is_outbuf_size_                 ) ) ;

	OPTX_CHECK( optixAccelBuild(
				optx_context_,
				0,
				&ois_options,
				&obi_things_,
				1,
				is_tmpbuf,
				is_buffer_sizes.tempSizeInBytes,
				is_outbuf_,
				is_outbuf_size_,
				is_handle,
				nullptr, 0
				) ) ;

	CUDA_CHECK( cudaFree( (void*) is_tmpbuf ) ) ;
}

void Scene::update( OptixTraversableHandle is_handle ) {
	// IAS options
	OptixAccelBuildOptions ois_options    = {} ;
	ois_options.buildFlags                = OPTIX_BUILD_FLAG_NONE ;
	ois_options.operation                 = OPTIX_BUILD_OPERATION_UPDATE ;

	OPTX_CHECK( optixAccelBuild(
				optx_context_,
				0,
				&ois_options,
				&obi_things_,
				1,
				is_updbuf_,
				is_updbuf_size_,
				is_outbuf_,
				is_outbuf_size_,
				&is_handle,
				nullptr, 0
				) ) ;
}

void Scene::free() noexcept ( false ) {
	if ( is_updbuf_ ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( is_updbuf_ ) ) ) ;
	if ( is_outbuf_ ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( is_outbuf_ ) ) ) ;
	if ( ises_      ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( ises_      ) ) ) ;
}
