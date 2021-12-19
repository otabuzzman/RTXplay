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

Scene::Scene( const OptixDeviceContext& optx_context ) : optx_context_( optx_context ), is_outbuf_( 0 ), is_updbuf_( 0 ), ises_( 0 ), obi_ises_( {} ) {
}

Scene::~Scene() noexcept ( false ) {
	for ( auto b : as_outbuf_ ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;
	for ( auto v : vces_ ) for ( auto b : v ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;
	for ( auto v : ices_ ) for ( auto b : v ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;
	free() ;
}

unsigned int Scene::add( Object& object ) {
	const unsigned int id = static_cast<unsigned int>( as_handle_.size() ) ;

	// object's build input structures
	const unsigned int object_size = static_cast<unsigned int>( object.size() ) ;
	std::vector<OptixBuildInput> obi_object( object_size ) ;

	// object's shapes V/I device buffers
	std::vector<CUdeviceptr> obj_vces( object_size ) ;
	std::vector<CUdeviceptr> obj_ices( object_size ) ;

	const unsigned int obi_object_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT } ;

	// one build input for each shape
	for ( unsigned int s = 0 ; object_size>s ; s++ ) {
		OptixBuildInput obi_shape = {} ;
		obi_shape.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES ;

		// object's shapes tuple
		float3*      shp_vces ;
		unsigned int shp_vces_size ;
		uint3*       shp_ices ;
		unsigned int shp_ices_size ;
		std::tie( shp_vces, shp_vces_size, shp_ices, shp_ices_size ) = object[s] ;

		CUdeviceptr vces = 0 ;
		copyDataToDevice<float3>( vces, shp_vces, shp_vces_size ) ;
		obj_vces[s] = vces ;
		obi_shape.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3 ;
		obi_shape.triangleArray.numVertices                 = shp_vces_size ;
		obi_shape.triangleArray.vertexBuffers               = &obj_vces[s] ;

		CUdeviceptr ices = 0 ;
		copyDataToDevice<uint3>( ices, shp_ices, shp_ices_size ) ;
		obj_ices[s] = ices ;
		obi_shape.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ;
		obi_shape.triangleArray.numIndexTriplets            = shp_ices_size ;
		obi_shape.triangleArray.indexBuffer                 = obj_ices[s] ;

		obi_shape.triangleArray.flags                       = &obi_object_flags[0] ;
		obi_shape.triangleArray.numSbtRecords               = 1 ; // number of SBT records in Hit Group section
		obi_shape.triangleArray.sbtIndexOffsetBuffer        = 0 ;
		obi_shape.triangleArray.sbtIndexOffsetSizeInBytes   = 0 ;
		obi_shape.triangleArray.sbtIndexOffsetStrideInBytes = 0 ;

		obi_object[s] = obi_shape ;
	}
	vces_.push_back( obj_vces ) ;
	ices_.push_back( obj_ices ) ;

	// GAS options
	OptixAccelBuildOptions oas_options = {} ;
	oas_options.buildFlags             = OPTIX_BUILD_FLAG_NONE ;
	oas_options.operation              = OPTIX_BUILD_OPERATION_BUILD ;

	// request acceleration structure buffer sizes from OptiX
	OptixAccelBufferSizes as_buffer_sizes ;
	OPTX_CHECK( optixAccelComputeMemoryUsage(
				optx_context_,
				&oas_options,
				obi_object.data(),
				object_size,
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
				obi_object.data(),
				object_size,
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
	const unsigned int id = static_cast<unsigned int>( is_ises_.size() ) ;

	// objects's instance structure
	OptixInstance ois_object     = {} ;
	ois_object.instanceId        = id ;
	ois_object.visibilityMask    = 255 ; // OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK

	ois_object.flags             = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;
	ois_object.sbtOffset         = static_cast<unsigned int>( things_.size() ) ;

	const float transform[12] = {
		1., 0., 0., 0.,
		0., 1., 0., 0.,
		0., 0., 1., 0.
	} ;
	memcpy( ois_object.transform, transform, sizeof( float )*12 ) ;

	ois_object.traversableHandle = as_handle_[object] ;

	is_ises_.push_back( ois_object ) ;

	const unsigned int object_size = static_cast<unsigned int>( vces_[object].size() ) ; 
	for ( unsigned int s = 0 ; object_size>s ; s++ ) {
		thing.vces = reinterpret_cast<float3*>( vces_[object][s] ) ;
		thing.ices = reinterpret_cast<uint3*> ( ices_[object][s] ) ;
		things_.push_back( thing ) ;
	}

	return id ;
}

bool Scene::set( unsigned int is_id, const float* transform ) {
	if ( is_ises_.size()>is_id ) {
		OptixInstance* is_instance = &is_ises_[is_id] ;
		memcpy( &is_instance->transform, transform, sizeof( float )*12 ) ;

		if ( ises_ ) { // update
			OptixInstance* instance = &reinterpret_cast<OptixInstance*>( ises_ )[is_id] ;
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

bool Scene::get( unsigned int is_id, float* transform ) {
	if ( is_ises_.size()>is_id ) {
		OptixInstance* is_instance = &is_ises_[is_id] ;
		memcpy( transform, &is_instance->transform, sizeof( float )*12 ) ;

		return true ;
	} else
		return false ;
}

void Scene::build( OptixTraversableHandle* is_handle ) {
	free() ;
	// instances build input structure
	copyDataToDevice<OptixInstance>( ises_, is_ises_.data(), static_cast<unsigned int>( is_ises_.size() ) ) ;
	obi_ises_.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	obi_ises_.instanceArray.instances    = ises_ ;
	obi_ises_.instanceArray.numInstances = static_cast<unsigned int>( is_ises_.size() ) ;

	// IAS options
	OptixAccelBuildOptions ois_options   = {} ;
	ois_options.buildFlags               = OPTIX_BUILD_FLAG_ALLOW_UPDATE ;
	ois_options.operation                = OPTIX_BUILD_OPERATION_BUILD ;

	// request acceleration structure buffer sizes from OptiX
	OptixAccelBufferSizes is_buffer_sizes ;
	OPTX_CHECK( optixAccelComputeMemoryUsage(
				optx_context_,
				&ois_options,
				&obi_ises_,
				1,
				&is_buffer_sizes
				) ) ;

	// allocate GPU memory for acceleration structure buffers
	CUdeviceptr is_tmpbuf ;
	is_updbuf_size_ = is_buffer_sizes.tempUpdateSizeInBytes ;
	is_outbuf_size_ = is_buffer_sizes.outputSizeInBytes ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &is_tmpbuf  ), is_buffer_sizes.tempSizeInBytes ) ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &is_updbuf_ ), is_updbuf_size_                 ) ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &is_outbuf_ ), is_outbuf_size_                 ) ) ;

	OPTX_CHECK( optixAccelBuild(
				optx_context_,
				0,
				&ois_options,
				&obi_ises_,
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
	ois_options.buildFlags                = OPTIX_BUILD_FLAG_ALLOW_UPDATE ;
	ois_options.operation                 = OPTIX_BUILD_OPERATION_UPDATE ;

	OPTX_CHECK( optixAccelBuild(
				optx_context_,
				0,
				&ois_options,
				&obi_ises_,
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

unsigned int Scene::size() {
	return static_cast<unsigned int>( things_.size() ) ;
}
