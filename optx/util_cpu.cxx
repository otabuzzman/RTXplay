#include <vector>
#include <memory>

#include <optix.h>
#include <optix_stubs.h>

#include <vector_functions.h>
#include <vector_types.h>

#include <sutil/Exception.h>

#include "camera.h"
#include "util.h"

#include "optixTriangle.h"

namespace util {

// PTX sources of shaders
extern "C" const char shader_all[] ;

// communicate state between optx* functions defined below
static struct {
	OptixDeviceContext          optx_context ;

	OptixTraversableHandle      as_handle ;
	CUdeviceptr                 d_as_outbuf ;
//	CUdeviceptr                 d_as_zipbuf ;

	OptixModule                 module_all              = nullptr ;
	OptixPipelineCompileOptions pipeline_cc_options     = {} ;

	OptixProgramGroup           program_group_camera    = nullptr ;
	OptixProgramGroup           program_group_ambient   = nullptr ;
	OptixProgramGroup           program_group_optics[3] = {} ;

	OptixPipeline               pipeline                = nullptr ;

	OptixShaderBindingTable     sbt                     = {} ;

	uchar4*                     d_image ;
} optx_state ;

void optxInitialize() noexcept( false ) {
	CUDA_CHECK( cudaFree( 0 ) ) ;
	OPTIX_CHECK( optixInit() ) ;

	OptixDeviceContextOptions optx_options = {} ;
	optx_options.logCallbackFunction       = &util::optxLogStderr ;
	optx_options.logCallbackLevel          = 4 ;

	// use current (0) CUDA context
	CUcontext cuda_context = 0 ;
	OPTIX_CHECK( optixDeviceContextCreate( cuda_context, &optx_options, &optx_state.optx_context ) ) ;
}

void optxBuildAccelerationStructure( const Things& things ) noexcept( false ) {
	// build input structures of things in scene
	std::vector<OptixBuildInput> obi_things ;
	obi_things.resize( things.size() ) ;

	// GPU pointers at vertices lists of things in scene
	std::vector<CUdeviceptr> d_vces ;
	d_vces.resize( things.size() ) ;

	// GPU pointers at lists of triangles lists of things in scene
	// each triangle made up of indexed vertices triplet
	std::vector<CUdeviceptr> d_ices ;
	d_ices.resize( things.size() ) ;

	// create build input strucure for each thing in scene
	for ( unsigned int i = 0 ; things.size()>i ; i++ ) {
		// copy this thing's vertices to GPU
		const std::vector<float3> vces = things[i]->vces() ;
		const size_t vces_size = sizeof( float3 )*vces.size() ;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vces[i] ), vces_size ) ) ;
		CUDA_CHECK( cudaMemcpy(
			reinterpret_cast<void*>( d_vces[i] ),
			vces.data(),
			vces_size,
			cudaMemcpyHostToDevice
			) ) ;

		// copy this thing's indices to GPU
		const std::vector<uint3> ices = things[i]->ices() ;
		const size_t ices_size = sizeof( uint3 )*ices.size() ;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_ices[i] ), ices_size ) ) ;
		CUDA_CHECK( cudaMemcpy(
			reinterpret_cast<void*>( d_ices[i] ),
			ices.data(),
			ices_size,
			cudaMemcpyHostToDevice
			) ) ;
		// setup this thing's build input structure
		OptixBuildInput obi_thing = {} ;
		obi_thing.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES ;

		obi_thing.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3 ;
		obi_thing.triangleArray.numVertices                 = static_cast<unsigned int>( vces.size() ) ;
		obi_thing.triangleArray.vertexBuffers               = &d_vces[i] ;

		obi_thing.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ;
		obi_thing.triangleArray.numIndexTriplets            = static_cast<unsigned int>( ices.size() ) ;
		obi_thing.triangleArray.indexBuffer                 = d_ices[i] ;

		const unsigned int obi_thing_flags[1]               = { OPTIX_GEOMETRY_FLAG_NONE } ;
		obi_thing.triangleArray.flags                       = obi_thing_flags ;
		obi_thing.triangleArray.numSbtRecords               = 1 ; // number of SBT records in Hit Group section
		obi_thing.triangleArray.sbtIndexOffsetBuffer        = 0 ;
		obi_thing.triangleArray.sbtIndexOffsetSizeInBytes   = 0 ;
		obi_thing.triangleArray.sbtIndexOffsetStrideInBytes = 0 ;

		obi_things[i] = obi_thing ;
	}

	OptixAccelBuildOptions oas_options = {} ;
	oas_options.buildFlags             = OPTIX_BUILD_FLAG_NONE ;
	oas_options.operation              = OPTIX_BUILD_OPERATION_BUILD ;

	// request acceleration structure buffer sizes from OptiX
	OptixAccelBufferSizes as_buffer_sizes ;
	OPTIX_CHECK( optixAccelComputeMemoryUsage(
				optx_state.optx_context,
				&oas_options,
				obi_things.data(),
				static_cast<unsigned int>( obi_things.size() ),
				&as_buffer_sizes
				) ) ;

	// allocate GPU memory for acceleration structure compaction buffer size
//	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_zipbuf_size ), sizeof( unsigned long long ) ) ) ;
//	CUDABuffer d_as_zipbuf_size ;

	// provide request description for acceleration structure compaction buffer size
//	OptixAccelEmitDesc oas_request ;
//	oas_request.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE ;
//	oas_request.result = d_as_zipbuf_size ;

	// allocate GPU memory for acceleration structure buffers
	CUdeviceptr d_as_tmpbuf ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_tmpbuf ), as_buffer_sizes.tempSizeInBytes ) ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &optx_state.d_as_outbuf ), as_buffer_sizes.outputSizeInBytes ) ) ;

	// build acceleration structure
	OPTIX_CHECK( optixAccelBuild(
				optx_state.optx_context,
				0,
				&oas_options,
				obi_things.data(),
				static_cast<unsigned int>( obi_things.size() ),
				d_as_tmpbuf,
				as_buffer_sizes.tempSizeInBytes,
				optx_state.d_as_outbuf,
				as_buffer_sizes.outputSizeInBytes,
				&optx_state.as_handle,
				nullptr, 0
//				&oas_request, 1
				) ) ;

	// retrieve acceleration structure compaction buffer size from GPU memory
//	unsigned long long as_zipbuf_size ;
//	CUDA_CHECK( cudaMemcpy(
//				reinterpret_cast<void*>( &as_zipbuf_size ),
//				d_as_zipbuf_size,
//				sizeof( unsigned long long ),
//				cudaMemcpyDeviceToHost
//				) ) ;

	// allocate GPU memory for acceleration structure compaction buffer
//	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &optx_state.d_as_zipbuf ), as_zipbuf_size ) ) ;
	// condense previously built acceleration structure
//	OPTIX_CHECK(optixAccelCompact(
//				optixContext,
//				0,
//				optx_state.as_handle,
//				optx_state.d_as_zipbuf,
//				d_as_zipbuf_size,
//				&optx_state.as_handle) ) ;

	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_tmpbuf ) ) ) ;
//	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( optx_state.d_as_outbuf ) ) ) ;
//	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_zipbuf_size ) ) ) ;
	for ( const CUdeviceptr p : d_vces ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( p ) ) ) ;
	for ( const CUdeviceptr p : d_ices ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( p ) ) ) ;
}

void optxCreateModules() noexcept( false ) {
	char   log[2048] ;
	size_t sizeof_log = sizeof( log ) ;

	OptixModuleCompileOptions module_cc_options = {} ;
	module_cc_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	module_cc_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_cc_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

	optx_state.pipeline_cc_options.usesMotionBlur                   = false ;
	optx_state.pipeline_cc_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS ;
	optx_state.pipeline_cc_options.numPayloadValues                 = 3 ;
	optx_state.pipeline_cc_options.numAttributeValues               = 3 ;
	optx_state.pipeline_cc_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE ;
	optx_state.pipeline_cc_options.pipelineLaunchParamsVariableName = "lp_general" ;
	optx_state.pipeline_cc_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ;

	// compile (each) shader source file into a module
	const size_t shader_all_size = strlen( &shader_all[0] ) ;
	OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
				optx_state.optx_context,
				&module_cc_options,
				&optx_state.pipeline_cc_options,
				&shader_all[0],
				shader_all_size,
				log,
				&sizeof_log,
				&optx_state.module_all
				) ) ;
}

void optxCreateProgramGroups() noexcept( false ) {
	char   log[2048] ;
	size_t sizeof_log = sizeof( log ) ;

	OptixProgramGroupOptions program_group_options = {} ;

	// Ray Generation program group
	OptixProgramGroupDesc program_group_camera_desc           = {} ;
	program_group_camera_desc.kind                            = OPTIX_PROGRAM_GROUP_KIND_RAYGEN ;
	program_group_camera_desc.raygen.module                   = optx_state.module_all;
	// function in shader source file with __global__ decorator
	program_group_camera_desc.raygen.entryFunctionName        = "__raygen__camera" ;
	OPTIX_CHECK_LOG( optixProgramGroupCreate(
				optx_state.optx_context,
				&program_group_camera_desc,
				1,
				&program_group_options,
				log,
				&sizeof_log,
				&optx_state.program_group_camera
				) ) ;

	// Miss program group
	OptixProgramGroupDesc program_group_ambient_desc   = {} ;

	program_group_ambient_desc.kind                           = OPTIX_PROGRAM_GROUP_KIND_MISS ;
	program_group_ambient_desc.miss.module                    = optx_state.module_all ;
	// function in shader source file with __global__ decorator
	program_group_ambient_desc.miss.entryFunctionName         = "__miss__ambient" ;
	OPTIX_CHECK_LOG( optixProgramGroupCreate(
				optx_state.optx_context,
				&program_group_ambient_desc,
				1,
				&program_group_options,
				log,
				&sizeof_log,
				&optx_state.program_group_ambient
				) ) ;

	// Hit Group program groups
	OptixProgramGroupDesc program_group_optics_desc[3] = {} ;
	// multiple program groups at once
	program_group_optics_desc[0].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP ;
	program_group_optics_desc[0].hitgroup.moduleCH            = optx_state.module_all;
	program_group_optics_desc[0].hitgroup.entryFunctionNameCH = "__closesthit__diffuse" ;
	program_group_optics_desc[1].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP ;
	program_group_optics_desc[1].hitgroup.moduleCH            = optx_state.module_all;
	program_group_optics_desc[1].hitgroup.entryFunctionNameCH = "__closesthit__reflect" ;
	program_group_optics_desc[2].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP ;
	program_group_optics_desc[2].hitgroup.moduleCH            = optx_state.module_all;
	program_group_optics_desc[2].hitgroup.entryFunctionNameCH = "__closesthit__refract" ;
	OPTIX_CHECK_LOG( optixProgramGroupCreate(
				optx_state.optx_context,
				&program_group_optics_desc[0],
				3,
				&program_group_options,
				log,
				&sizeof_log,
				&optx_state.program_group_optics[0]
				) ) ;
}

void optxLinkPipeline() noexcept( false ) {
	char   log[2048] ;
	size_t sizeof_log = sizeof( log ) ;

	const uint32_t    max_trace_depth  = 1 ;
	OptixProgramGroup program_groups[] = {
		optx_state.program_group_camera,
		optx_state.program_group_ambient,
		optx_state.program_group_optics[0],
		optx_state.program_group_optics[1],
		optx_state.program_group_optics[2] } ;

	OptixPipelineLinkOptions pipeline_ld_options = {} ;
	pipeline_ld_options.maxTraceDepth            = max_trace_depth ;
	pipeline_ld_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL ;
	OPTIX_CHECK_LOG( optixPipelineCreate(
				optx_state.optx_context,
				&optx_state.pipeline_cc_options,
				&pipeline_ld_options,
				program_groups,
				sizeof( program_groups )/sizeof( program_groups[0] ),
				log,
				&sizeof_log,
				&optx_state.pipeline
				) ) ;

	OPTIX_CHECK( optixPipelineSetStackSize(
				optx_state.pipeline,
				2*1024, // direct callable stack size (called from AH and IS programs)
				2*1024, // direct callable stack size (called from RG, MS and CH programs)
				2*1024, // continuation callable stack size
				1       // maxTraversableGraphDepth (acceleration structure depth)
				) ) ;
}

void optxBuildShaderBindingTable( const Things& things ) noexcept( false ) {
	SbtRecordRG sbt_record_camera ;
	// Ray Generation program group SBT record header
	OPTIX_CHECK( optixSbtRecordPackHeader( optx_state.program_group_camera, &sbt_record_camera ) ) ;
	// copy SBT record to GPU
	CUdeviceptr  d_sbt_record_camera ;
	const size_t sbt_record_camera_size = sizeof( SbtRecordRG ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_record_camera ), sbt_record_camera_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( d_sbt_record_camera ),
				&sbt_record_camera,
				sbt_record_camera_size,
				cudaMemcpyHostToDevice
				) ) ;
	// set SBT Ray Generation section to point at record
	optx_state.sbt.raygenRecord = d_sbt_record_camera ;

	SbtRecordMS sbt_record_ambient ;
	// Miss program group SBT record header
	OPTIX_CHECK( optixSbtRecordPackHeader( optx_state.program_group_ambient, &sbt_record_ambient ) ) ;
	// copy SBT record to GPU
	CUdeviceptr d_sbt_record_ambient ;
	const size_t sbt_record_ambient_size = sizeof( SbtRecordMS ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_record_ambient ), sbt_record_ambient_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( d_sbt_record_ambient ),
				&sbt_record_ambient,
				sbt_record_ambient_size,
				cudaMemcpyHostToDevice
				) ) ;
	// set SBT Miss section to point at record(s)
	optx_state.sbt.missRecordBase          = d_sbt_record_ambient ;
	// set size and number of Miss records
	optx_state.sbt.missRecordStrideInBytes = sizeof( SbtRecordMS ) ;
	optx_state.sbt.missRecordCount         = 1 ;

	// SBT Record buffer for Hit Group program groups
	std::vector<SbtRecordHG> sbt_record_buffer ;
	sbt_record_buffer.resize( things.size() ) ;

	// set SBT record for each thing in scene
	for ( unsigned int i = 0 ; things.size()>i ; i++ ) {
		// this thing's SBT record
		SbtRecordHG sbt_record_optics ;
		sbt_record_optics.data = things[i]->optics() ;
		// setup SBT record header
		OPTIX_CHECK( optixSbtRecordPackHeader( optx_state.program_group_optics[0], &sbt_record_optics ) ) ;
		// save thing's SBT Record to buffer
		sbt_record_buffer[i] = sbt_record_optics ;
	}

	// copy SBT record to GPU
	CUdeviceptr d_sbt_record_buffer ;
	const size_t sbt_record_buffer_size = sizeof( SbtRecordHG )*sbt_record_buffer.size() ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_record_buffer ), sbt_record_buffer_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( d_sbt_record_buffer ),
				sbt_record_buffer.data(),
				sbt_record_buffer_size,
				cudaMemcpyHostToDevice
				) ) ;
	// set SBT Hit Group section to point at records
	optx_state.sbt.hitgroupRecordBase          = d_sbt_record_buffer ;
	// set size and number of Hit Group records
	optx_state.sbt.hitgroupRecordStrideInBytes = sizeof( SbtRecordHG ) ;
	optx_state.sbt.hitgroupRecordCount         = static_cast<unsigned int>( sbt_record_buffer.size() ) ;
}

const std::vector<uchar4> optxLaunchPipeline( const int w, const int h ) {
	std::vector<uchar4> image ;

	CUstream cuda_stream ;
	CUDA_CHECK( cudaStreamCreate( &cuda_stream ) ) ;

	CUDA_CHECK( cudaMalloc(
				reinterpret_cast<void**>( &optx_state.d_image ),
				w*h*sizeof( uchar4 )
				) ) ;

	LpGeneral lp_general ; // launch parameter

	lp_general.image     = optx_state.d_image ;
	lp_general.image_w   = w ;
	lp_general.image_h   = h ;

	lp_general.as_handle = optx_state.as_handle ;

	CUdeviceptr d_lp_general ;
	const size_t lp_general_size = sizeof( LpGeneral ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_lp_general ), lp_general_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( d_lp_general ),
				&lp_general,
				lp_general_size,
				cudaMemcpyHostToDevice
				) ) ;

	OPTIX_CHECK( optixLaunch(
				optx_state.pipeline,
				cuda_stream,
				d_lp_general,
				lp_general_size,
				&optx_state.sbt,
				w/*x*/, h/*y*/, 1/*z*/ ) ) ;
	CUDA_SYNC_CHECK() ;

	image.resize( w*h ) ;
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( image.data() ),
				optx_state.d_image,
				w*h*sizeof( uchar4 ),
				cudaMemcpyDeviceToHost
				) ) ;

	return image ;
}

void optxCleanup() noexcept( false ) {
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( optx_state.sbt.raygenRecord       ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( optx_state.sbt.missRecordBase     ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( optx_state.sbt.hitgroupRecordBase ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( optx_state.d_as_outbuf            ) ) ) ;
//	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( optx_state.d_as_zipbuf            ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( optx_state.d_image                ) ) ) ;

	OPTIX_CHECK( optixPipelineDestroy    ( optx_state.pipeline                ) ) ;
	OPTIX_CHECK( optixProgramGroupDestroy( optx_state.program_group_optics[0] ) ) ;
	OPTIX_CHECK( optixProgramGroupDestroy( optx_state.program_group_optics[1] ) ) ;
	OPTIX_CHECK( optixProgramGroupDestroy( optx_state.program_group_optics[2] ) ) ;
	OPTIX_CHECK( optixProgramGroupDestroy( optx_state.program_group_ambient   ) ) ;
	OPTIX_CHECK( optixProgramGroupDestroy( optx_state.program_group_camera    ) ) ;
	OPTIX_CHECK( optixModuleDestroy      ( optx_state.module_all              ) ) ;

	OPTIX_CHECK( optixDeviceContextDestroy( optx_state.optx_context ) ) ;
}

}
