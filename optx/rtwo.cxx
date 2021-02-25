#include <array>
#include <chrono>
#include <string>
#include <vector>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <sutil/Exception.h>

#include <vector_functions.h>
#include <vector_types.h>

#include "camera.h"
#include "optics.h"
#include "sphere.h"
#include "things.h"
#include "util.h"
#include "v.h"

#include "rtwo.h"

using V::operator- ;
using V::operator* ;

// PTX sources of shaders
extern "C" const char camera_ptx[] ;
extern "C" const char optics_ptx[] ;

const Things scene() {
	Optics o ;
	Things s ;

	o.type = OPTICS_TYPE_DIFFUSE ;
	o.diffuse.albedo = { .5f, .5f, .5f } ;
	s.push_back( std::make_shared<Sphere>( make_float3( 0.f, -1000.f, 0.f ), 1000.f, o, false, 9 ) ) ;

	for ( int a = -11 ; a<11 ; a++ ) {
		for ( int b = -11 ; b<11 ; b++ ) {
			auto bbox = false ; // .3f>util::rnd() ? true : false ;
			auto select = util::rnd() ;
			float3 center = make_float3( a+.9f*util::rnd(), .2f, b+.9f*util::rnd() ) ;
			if ( V::len( center-make_float3( 4.f, .2f, 0.f ) )>.9f ) {
				if ( select<.8f ) {
					o.type = OPTICS_TYPE_DIFFUSE ;
					o.diffuse.albedo = V::rnd()*V::rnd() ;
					s.push_back( std::make_shared<Sphere>( center, .2f, o, bbox ) ) ;
				} else if ( select<.95f ) {
					o.type = OPTICS_TYPE_REFLECT ;
					o.reflect.albedo = V::rnd( .5f, 1.f ) ;
					o.reflect.fuzz = util::rnd( 0.f, .5f ) ;
					s.push_back( std::make_shared<Sphere>( center, .2f, o, bbox ) ) ;
				} else {
					o.type = OPTICS_TYPE_REFRACT ;
					o.refract.index = 1.5f ;
					s.push_back( std::make_shared<Sphere>( center, .2f, o, bbox, 3 ) ) ;
				}
			}
		}
	}

	o.type = OPTICS_TYPE_REFRACT ;
	o.refract.index  = 1.5f ;
	s.push_back( std::make_shared<Sphere>( make_float3(  0.f, 1.f, 0.f ), 1.f, o, false, 8 ) ) ;
	o.type = OPTICS_TYPE_DIFFUSE ;
	o.diffuse.albedo = { .4f, .2f, .1f } ;
	s.push_back( std::make_shared<Sphere>( make_float3( -4.f, 1.f, 0.f ), 1.f, o ) ) ;
	o.type = OPTICS_TYPE_REFLECT ;
	o.reflect.albedo = { .7f, .6f, .5f } ;
	o.reflect.fuzz   = 0.f ;
	s.push_back( std::make_shared<Sphere>( make_float3(  4.f, 1.f, 0.f ), 1.f, o, false, 3 ) ) ;

	return s ;
}

int main() {
	Things things = scene() ;

	float aspratio = 3.f/2.f ;

	SbtRecordRG sbt_record_camera ;
	sbt_record_camera.data.set(
		{13.f, 2.f, 3.f} /*eye*/,
		{ 0.f, 0.f, 0.f} /*pat*/,
		{ 0.f, 1.f, 0.f} /*vup*/,
		20.f /*fov*/,
		aspratio,
		.1f  /*aperture*/,
		10.f /*distance*/ ) ;

	const int w = 1200 ;                           // image width in pixels
	const int h = static_cast<int>( w/aspratio ) ; // image height in pixels
	const int spp = 500 ;                          // samples per pixel
	const int depth = 50 ;                         // recursion depth

	SbtRecordMS sbt_record_ambient ;
	sbt_record_ambient.data = { .5f, .7f, 1.f } ;

	try {
		// initialize
		OptixDeviceContext optx_context = nullptr ;
		{
			CUDA_CHECK( cudaFree( 0 ) ) ;
			OPTIX_CHECK( optixInit() ) ;

			OptixDeviceContextOptions optx_options = {} ;
			optx_options.logCallbackFunction       = &util::optxLogStderr ;
			optx_options.logCallbackLevel          = 4 ;

			// use current (0) CUDA context
			CUcontext cuda_context = 0 ;
			OPTIX_CHECK( optixDeviceContextCreate( cuda_context, &optx_options, &optx_context ) ) ;
		}



		// build acceleration structure
		OptixTraversableHandle   as_handle ;
		CUdeviceptr              d_as_outbuf ;
		// acceleration structure compaction buffer
//		CUdeviceptr              d_as_zipbuf ;
		{
			// build input structures of things in scene
			std::vector<OptixBuildInput> obi_things ;
			obi_things.resize( things.size() ) ;

			// GPU pointers at vertices lists of things in scene
			std::vector<CUdeviceptr> d_vces ;
			d_vces.resize( things.size() ) ;

			// GPU pointers at triangles lists of things in scene
			std::vector<CUdeviceptr> d_ices ;
			d_ices.resize( things.size() ) ;

			// create build input strucure for each thing in scene
			for ( unsigned int i = 0 ; things.size()>i ; i++ ) {
				d_vces[i] = reinterpret_cast<CUdeviceptr>( things[i]->d_vces() ) ;
				d_ices[i] = reinterpret_cast<CUdeviceptr>( things[i]->d_ices() ) ;
				// setup this thing's build input structure
				OptixBuildInput obi_thing = {} ;
				obi_thing.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES ;

				obi_thing.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3 ;
				obi_thing.triangleArray.numVertices                 = things[i]->num_vces() ;
				obi_thing.triangleArray.vertexBuffers               = &d_vces[i] ;

				obi_thing.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ;
				obi_thing.triangleArray.numIndexTriplets            = things[i]->num_ices() ;
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
						optx_context,
						&oas_options,
						obi_things.data(),
						static_cast<unsigned int>( obi_things.size() ),
						&as_buffer_sizes
						) ) ;

			// allocate GPU memory for acceleration structure compaction buffer size
//			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_zipbuf_size ), sizeof( unsigned long long ) ) ) ;
//			CUDABuffer d_as_zipbuf_size ;

			// provide request description for acceleration structure compaction buffer size
//			OptixAccelEmitDesc oas_request ;
//			oas_request.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE ;
//			oas_request.result = d_as_zipbuf_size ;

			// allocate GPU memory for acceleration structure buffers
			CUdeviceptr d_as_tmpbuf ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_tmpbuf ), as_buffer_sizes.tempSizeInBytes ) ) ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_outbuf ), as_buffer_sizes.outputSizeInBytes ) ) ;

			// build acceleration structure
			OPTIX_CHECK( optixAccelBuild(
						optx_context,
						0,
						&oas_options,
						obi_things.data(),
						static_cast<unsigned int>( obi_things.size() ),
						d_as_tmpbuf,
						as_buffer_sizes.tempSizeInBytes,
						d_as_outbuf,
						as_buffer_sizes.outputSizeInBytes,
						&as_handle,
						nullptr, 0
						// acceleration structure compaction (must comment previous line)
//						&oas_request, 1
						) ) ;

			// retrieve acceleration structure compaction buffer size from GPU memory
//			unsigned long long as_zipbuf_size ;
//			CUDA_CHECK( cudaMemcpy(
//						reinterpret_cast<void*>( &as_zipbuf_size ),
//						d_as_zipbuf_size,
//						sizeof( unsigned long long ),
//						cudaMemcpyDeviceToHost
//						) ) ;

			// allocate GPU memory for acceleration structure compaction buffer
//			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_zipbuf ), as_zipbuf_size ) ) ;
			// condense previously built acceleration structure
//			OPTIX_CHECK(optixAccelCompact(
//						optixContext,
//						0,
//						as_handle,
//						d_as_zipbuf,
//						d_as_zipbuf_size,
//						&as_handle) ) ;

			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_tmpbuf ) ) ) ;
			// free GPU memory for acceleration structure compaction buffer
//			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_outbuf ) ) ) ;
//			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_zipbuf_size ) ) ) ;
		}



		// create module(s)
		OptixModule module_camera = nullptr ;
		OptixModule module_optics = nullptr ;
		OptixPipelineCompileOptions pipeline_cc_options = {} ;
		{
			char   log[2048] ;
			size_t sizeof_log = sizeof( log ) ;

			OptixModuleCompileOptions module_cc_options = {} ;
			module_cc_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT ;
			module_cc_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT ;
			module_cc_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_FULL ;

			pipeline_cc_options.usesMotionBlur                   = false ;
			pipeline_cc_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS ;
			pipeline_cc_options.numPayloadValues                 = 6 ; // R, G, B, RNG (2x), depth
			pipeline_cc_options.numAttributeValues               = 2 ;
			pipeline_cc_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE ;
			pipeline_cc_options.pipelineLaunchParamsVariableName = "lp_general" ;
			pipeline_cc_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ;

			// compile (each) shader source file into a module
			const size_t camera_ptx_size = strlen( &camera_ptx[0] ) ;
			OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
						optx_context,
						&module_cc_options,
						&pipeline_cc_options,
						&camera_ptx[0],
						camera_ptx_size,
						log,
						&sizeof_log,
						&module_camera
						) ) ;

			const size_t optics_ptx_size = strlen( &optics_ptx[0] ) ;
			OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
						optx_context,
						&module_cc_options,
						&pipeline_cc_options,
						&optics_ptx[0],
						optics_ptx_size,
						log,
						&sizeof_log,
						&module_optics
						) ) ;
		}



		// create program groups
		OptixProgramGroup program_group_camera    = nullptr ;
		OptixProgramGroup program_group_ambient   = nullptr ;
		OptixProgramGroup program_group_optics[OPTICS_TYPE_NUM] = {} ;
		{
			char   log[2048] ;
			size_t sizeof_log = sizeof( log ) ;

			OptixProgramGroupOptions program_group_options = {} ;

			// Ray Generation program group
			OptixProgramGroupDesc program_group_camera_desc           = {} ;
			program_group_camera_desc.kind                            = OPTIX_PROGRAM_GROUP_KIND_RAYGEN ;
			program_group_camera_desc.raygen.module                   = module_camera ;
			// function in shader source file with __global__ decorator
			program_group_camera_desc.raygen.entryFunctionName        = "__raygen__camera" ;
			OPTIX_CHECK_LOG( optixProgramGroupCreate(
						optx_context,
						&program_group_camera_desc,
						1,
						&program_group_options,
						log,
						&sizeof_log,
						&program_group_camera
						) ) ;

			// Miss program group
			OptixProgramGroupDesc program_group_ambient_desc   = {} ;

			program_group_ambient_desc.kind                           = OPTIX_PROGRAM_GROUP_KIND_MISS ;
			program_group_ambient_desc.miss.module                    = module_camera ;
			// function in shader source file with __global__ decorator
			program_group_ambient_desc.miss.entryFunctionName         = "__miss__ambient" ;
			OPTIX_CHECK_LOG( optixProgramGroupCreate(
						optx_context,
						&program_group_ambient_desc,
						1,
						&program_group_options,
						log,
						&sizeof_log,
						&program_group_ambient
						) ) ;

			// Hit Group program groups
			OptixProgramGroupDesc program_group_optics_desc[3] = {} ;
			// multiple program groups at once
			program_group_optics_desc[OPTICS_TYPE_DIFFUSE].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP ;
			program_group_optics_desc[OPTICS_TYPE_DIFFUSE].hitgroup.entryFunctionNameCH = "__closesthit__diffuse" ;
			program_group_optics_desc[OPTICS_TYPE_DIFFUSE].hitgroup.moduleCH            = module_optics ;
			program_group_optics_desc[OPTICS_TYPE_REFLECT].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP ;
			program_group_optics_desc[OPTICS_TYPE_REFLECT].hitgroup.moduleCH            = module_optics ;
			program_group_optics_desc[OPTICS_TYPE_REFLECT].hitgroup.entryFunctionNameCH = "__closesthit__reflect" ;
			program_group_optics_desc[OPTICS_TYPE_REFRACT].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP ;
			program_group_optics_desc[OPTICS_TYPE_REFRACT].hitgroup.moduleCH            = module_optics ;
			program_group_optics_desc[OPTICS_TYPE_REFRACT].hitgroup.entryFunctionNameCH = "__closesthit__refract" ;
			OPTIX_CHECK_LOG( optixProgramGroupCreate(
						optx_context,
						&program_group_optics_desc[0],
						3,
						&program_group_options,
						log,
						&sizeof_log,
						&program_group_optics[0]
						) ) ;
		}



		// link pipeline
		OptixPipeline pipeline = nullptr ;
		{
			char   log[2048] ;
			size_t sizeof_log = sizeof( log ) ;

			const unsigned int max_trace_depth  = 4 ;
			OptixProgramGroup program_groups[]  = {
				program_group_camera,
				program_group_ambient,
				program_group_optics[OPTICS_TYPE_DIFFUSE],
				program_group_optics[OPTICS_TYPE_REFLECT],
				program_group_optics[OPTICS_TYPE_REFRACT] } ;

			OptixPipelineLinkOptions pipeline_ld_options = {} ;
			pipeline_ld_options.maxTraceDepth            = max_trace_depth ;
			pipeline_ld_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL ;
			OPTIX_CHECK_LOG( optixPipelineCreate(
						optx_context,
						&pipeline_cc_options,
						&pipeline_ld_options,
						program_groups,
						sizeof( program_groups )/sizeof( program_groups[0] ),
						log,
						&sizeof_log,
						&pipeline
						) ) ;



			/*** calculate stack sizes

			OptixStackSizes stack_sizes = {} ;
			for( auto& prog_group : program_groups )
				OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) ) ;
//			fprintf( stderr, "cssRG: %u, cssMS: %u, cssCH: %u, cssAH: %u, cssIS: %u, cssCC: %u, dssDG: %u\n",
//				stack_sizes.cssRG,
//				stack_sizes.cssMS,
//				stack_sizes.cssCH,
//				stack_sizes.cssAH,
//				stack_sizes.cssIS,
//				stack_sizes.cssCC,
//				stack_sizes.dssDC ) ;

			unsigned int dssTrav ;
			unsigned int dssStat ;
			unsigned int css ;
			OPTIX_CHECK( optixUtilComputeStackSizes(
						&stack_sizes,
						max_trace_depth,
						0, // maxCCDepth
						0, // maxDCDEpth
						&dssTrav,
						&dssStat,
						&css ) ) ;
//			fprintf( stderr, "dss Traversal (IS, AH): %u, dss State (RG, MS, CH): %u, css: %u\n",
//				dssTrav,
//				dssStat,
//				css ) ;

			// max_trace_depth = 4 :
			// cssRG: 6496, cssMS: 0, cssCH: 32, cssAH: 0, cssIS: 0, cssCC: 0, dssDG: 0
			// dss Traversal (IS, AH): 0, dss State (RG, MS, CH): 0, css: 6624

			// max_trace_depth = 16 :
			// cssRG: 6496, cssMS: 0, cssCH: 32, cssAH: 0, cssIS: 0, cssCC: 0, dssDG: 0
			// dss Traversal (IS, AH): 0, dss State (RG, MS, CH): 0, css: 7008

			OPTIX_CHECK( optixPipelineSetStackSize(
						pipeline,
						dssTrav, // direct callable stack size (called from AH and IS programs)
						dssStat, // direct callable stack size (called from RG, MS and CH programs)
						css,     // continuation callable stack size
						1        // maxTraversableGraphDepth (acceleration structure depth)
						) ) ;

			***/



			// comment if using code from comment block titled `calculate stack sizes´
			OPTIX_CHECK( optixPipelineSetStackSize(
						pipeline,
						4*1024, // direct callable stack size (called from AH and IS programs)
						4*1024, // direct callable stack size (called from RG, MS and CH programs)
						4*1024, // continuation callable stack size
						1       // maxTraversableGraphDepth (acceleration structure depth)
						) ) ;
		}



		// build shader binding table
		OptixShaderBindingTable sbt = {} ;
		{
			// Ray Generation program group SBT record header
			OPTIX_CHECK( optixSbtRecordPackHeader( program_group_camera, &sbt_record_camera ) ) ;
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
			sbt.raygenRecord = d_sbt_record_camera ;

			// Miss program group SBT record header
			OPTIX_CHECK( optixSbtRecordPackHeader( program_group_ambient, &sbt_record_ambient ) ) ;
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
			sbt.missRecordBase          = d_sbt_record_ambient ;
			// set size and number of Miss records
			sbt.missRecordStrideInBytes = sizeof( SbtRecordMS ) ;
			sbt.missRecordCount         = 1 ;

			// SBT Record buffer for Hit Group program groups
			std::vector<SbtRecordHG> sbt_record_buffer ;
			sbt_record_buffer.resize( things.size() ) ;

			// set SBT record for each thing in scene
			for ( unsigned int i = 0 ; things.size()>i ; i++ ) {
				// this thing's SBT record
				SbtRecordHG sbt_record_thing ;
				sbt_record_thing.data = *things[i] ;
				// setup SBT record header
				OPTIX_CHECK( optixSbtRecordPackHeader( program_group_optics[things[i]->optics().type], &sbt_record_thing ) ) ;
				// save thing's SBT Record to buffer
				sbt_record_buffer[i] = sbt_record_thing ;
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
			sbt.hitgroupRecordBase          = d_sbt_record_buffer ;
			// set size and number of Hit Group records
			sbt.hitgroupRecordStrideInBytes = sizeof( SbtRecordHG ) ;
			sbt.hitgroupRecordCount         = static_cast<unsigned int>( sbt_record_buffer.size() ) ;
		}



		auto t0 = std::chrono::high_resolution_clock::now() ;
		// launch pipeline
		LpGeneral lp_general ;
		{
			CUstream cuda_stream ;
			CUDA_CHECK( cudaStreamCreate( &cuda_stream ) ) ;

			CUDA_CHECK( cudaMalloc(
						reinterpret_cast<void**>( &lp_general.image ),
						w*h*sizeof( uchar4 )
						) ) ;
			lp_general.image_w   = w ;
			lp_general.image_h   = h ;

			lp_general.spp       = spp ;

			lp_general.depth     = depth ;

			lp_general.as_handle = as_handle ;

			CUdeviceptr d_lp_general ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_lp_general ), sizeof( LpGeneral ) ) ) ;
			CUDA_CHECK( cudaMemcpy(
						reinterpret_cast<void*>( d_lp_general ),
						&lp_general, sizeof( lp_general ),
						cudaMemcpyHostToDevice
						) ) ;

			const size_t lp_general_size = sizeof( LpGeneral ) ;
			OPTIX_CHECK( optixLaunch(
						pipeline,
						cuda_stream,
						d_lp_general,
						lp_general_size,
						&sbt,
						w/*x*/, h/*y*/, 1/*z*/ ) ) ;
			CUDA_SYNC_CHECK() ;
		}
		auto t1 = std::chrono::high_resolution_clock::now() ;
		long long int dt = std::chrono::duration_cast<std::chrono::milliseconds>( t1-t0 ).count() ;
		fprintf( stderr, "OptiX pipeline for RTWO ran %lld milliseconds\n", dt ) ;



		// output image
		std::vector<uchar4> image ;
		{
			image.resize( w*h ) ;
			CUDA_CHECK( cudaMemcpy(
						reinterpret_cast<void*>( image.data() ),
						lp_general.image,
						w*h*sizeof( uchar4 ),
						cudaMemcpyDeviceToHost
						) ) ;

			std::cout
				<< "P3\n" // magic PPM header
				<< w << ' ' << h << '\n' << 255 << '\n' ;

			for ( int y = h-1 ; y>=0 ; --y ) {
				for ( int x = 0 ; x<w ; ++x ) {
					auto p = image.data()[w*y+x] ;
					std::cout
						<< static_cast<int>( p.x ) << ' '
						<< static_cast<int>( p.y ) << ' '
						<< static_cast<int>( p.z ) << '\n' ;
				}
			}
		}



		// cleanup
		{
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_outbuf            ) ) ) ;
//			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_zipbuf            ) ) ) ;
			for ( unsigned int i = 0 ; things.size()>i ; i++ ) things[i] = nullptr ; // force thing's dtor
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.image       ) ) ) ;

			OPTIX_CHECK( optixPipelineDestroy    ( pipeline                ) ) ;
			OPTIX_CHECK( optixProgramGroupDestroy( program_group_optics[OPTICS_TYPE_DIFFUSE] ) ) ;
			OPTIX_CHECK( optixProgramGroupDestroy( program_group_optics[OPTICS_TYPE_REFLECT] ) ) ;
			OPTIX_CHECK( optixProgramGroupDestroy( program_group_optics[OPTICS_TYPE_REFRACT] ) ) ;
			OPTIX_CHECK( optixProgramGroupDestroy( program_group_ambient   ) ) ;
			OPTIX_CHECK( optixProgramGroupDestroy( program_group_camera    ) ) ;
			OPTIX_CHECK( optixModuleDestroy      ( module_camera              ) ) ;
			OPTIX_CHECK( optixModuleDestroy      ( module_optics              ) ) ;

			OPTIX_CHECK( optixDeviceContextDestroy( optx_context ) ) ;
		}
	} catch( std::exception& e ) {
		std::cerr << "exception: " << e.what() << "\n" ;

		return 1 ;
	}

	return 0 ;
}
