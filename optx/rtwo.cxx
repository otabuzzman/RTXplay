#include <array>
#include <chrono>
#include <string>
#include <vector>

#include <optix.h>
#include <optix_stubs.h>
/*** calculate stack sizes
#include <optix_stack_size.h>
***/
#include <optix_function_table_definition.h>

#include <vector_functions.h>
#include <vector_types.h>

#include <GLFW/glfw3.h>

#include "camera.h"
#include "optics.h"
#include "simpleui.h"
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

	LpGeneral lp_general ;
	lp_general.camera.set(
		{ 13.f, 2.f, 3.f } /*eye*/,
		{  0.f, 0.f, 0.f } /*pat*/,
		{  0.f, 1.f, 0.f } /*vup*/,
		20.f /*fov*/,
		aspratio,
		.1f  /*aperture*/,
		20.f /*distance*/ ) ;

	lp_general.image_w = 1200 ;                                            // image width in pixels
	lp_general.image_h = static_cast<int>( lp_general.image_w/aspratio ) ; // image height in pixels
	lp_general.spp = 50 ;                                                  // samples per pixel
	lp_general.depth = 16 ;                                                // recursion depth

	SbtRecordMS sbt_record_ambient ;
	sbt_record_ambient.data = { .5f, .7f, 1.f } ;

	try {
		// initialize
		OptixDeviceContext optx_context = nullptr ;
		{
			CUDA_CHECK( cudaFree( 0 ) ) ;
			OPTX_CHECK( optixInit() ) ;

			OptixDeviceContextOptions optx_options = {} ;
			optx_options.logCallbackFunction       = &util::optxLogStderr ;
			optx_options.logCallbackLevel          = 4 ;

			// use current (0) CUDA context
			CUcontext cuda_context = 0 ;
			OPTX_CHECK( optixDeviceContextCreate( cuda_context, &optx_options, &optx_context ) ) ;
		}



		// build acceleration structure
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
			OPTX_CHECK( optixAccelComputeMemoryUsage(
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
			OPTX_CHECK( optixAccelBuild(
						optx_context,
						0,
						&oas_options,
						obi_things.data(),
						static_cast<unsigned int>( obi_things.size() ),
						d_as_tmpbuf,
						as_buffer_sizes.tempSizeInBytes,
						d_as_outbuf,
						as_buffer_sizes.outputSizeInBytes,
						&lp_general.as_handle,
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
//			OPTX_CHECK( optixAccelCompact(
//						optixContext,
//						0,
//						lp_general.as_handle,
//						d_as_zipbuf,
//						d_as_zipbuf_size,
//						&lp_general.as_handle) ) ;

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
			OptixModuleCompileOptions module_cc_options = {} ;
			module_cc_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT ;
			module_cc_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 ; // OPTIX_COMPILE_OPTIMIZATION_DEFAULT
			module_cc_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_FULL ;     // OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT

			pipeline_cc_options.usesMotionBlur                   = false ;
			pipeline_cc_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS ;
			pipeline_cc_options.numPayloadValues                 = 6 ; // R, G, B, RNG (2x), depth
			pipeline_cc_options.numAttributeValues               = 2 ;
			pipeline_cc_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE ;
			pipeline_cc_options.pipelineLaunchParamsVariableName = "lp_general" ;
			pipeline_cc_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ;

			// compile (each) shader source file into a module
			const size_t camera_ptx_size = strlen( &camera_ptx[0] ) ;
			OPTX_CHECK_LOG( optixModuleCreateFromPTX(
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
			OPTX_CHECK_LOG( optixModuleCreateFromPTX(
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
			OptixProgramGroupOptions program_group_options = {} ;

			// Ray Generation program group
			OptixProgramGroupDesc program_group_camera_desc           = {} ;
			program_group_camera_desc.kind                            = OPTIX_PROGRAM_GROUP_KIND_RAYGEN ;
			program_group_camera_desc.raygen.module                   = module_camera ;
			// function in shader source file with __global__ decorator
			program_group_camera_desc.raygen.entryFunctionName        = "__raygen__camera" ;
			OPTX_CHECK_LOG( optixProgramGroupCreate(
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
			OPTX_CHECK_LOG( optixProgramGroupCreate(
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
			OPTX_CHECK_LOG( optixProgramGroupCreate(
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
			const unsigned int max_trace_depth  = lp_general.depth ;
			OptixProgramGroup program_groups[]  = {
				program_group_camera,
				program_group_ambient,
				program_group_optics[OPTICS_TYPE_DIFFUSE],
				program_group_optics[OPTICS_TYPE_REFLECT],
				program_group_optics[OPTICS_TYPE_REFRACT] } ;

			OptixPipelineLinkOptions pipeline_ld_options = {} ;
			pipeline_ld_options.maxTraceDepth            = max_trace_depth ;
			pipeline_ld_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL ;
			OPTX_CHECK_LOG( optixPipelineCreate(
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
			for ( auto& prog_group : program_groups )
				OPTX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) ) ;
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
			OPTX_CHECK( optixUtilComputeStackSizes(
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

			OPTX_CHECK( optixPipelineSetStackSize(
						pipeline,
						dssTrav, // direct callable stack size (called from AH and IS programs)
						dssStat, // direct callable stack size (called from RG, MS and CH programs)
						css,     // continuation callable stack size
						1        // maxTraversableGraphDepth (acceleration structure depth)
						) ) ;

			***/



			// comment if using code from comment block titled `calculate stack sizes´
			OPTX_CHECK( optixPipelineSetStackSize(
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
			SbtRecordRG sbt_record_nodata ;
			OPTX_CHECK( optixSbtRecordPackHeader( program_group_camera, &sbt_record_nodata ) ) ;
			// copy SBT record to GPU
			CUdeviceptr  d_sbt_record_nodata ;
			const size_t sbt_record_nodata_size = sizeof( SbtRecordRG ) ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_record_nodata ), sbt_record_nodata_size ) ) ;
			CUDA_CHECK( cudaMemcpy(
						reinterpret_cast<void*>( d_sbt_record_nodata ),
						&sbt_record_nodata,
						sbt_record_nodata_size,
						cudaMemcpyHostToDevice
						) ) ;
			// set SBT Ray Generation section to point at record
			sbt.raygenRecord = d_sbt_record_nodata ;

			// Miss program group SBT record header
			OPTX_CHECK( optixSbtRecordPackHeader( program_group_ambient, &sbt_record_ambient ) ) ;
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
				OPTX_CHECK( optixSbtRecordPackHeader( program_group_optics[things[i]->optics().type], &sbt_record_thing ) ) ;
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



		int current_dev, display_dev ;
		// check if X server executes on current device (SO #17994896)
		// if true current is display device as required for GL interop
		CUDA_CHECK( cudaGetDevice( &current_dev ) ) ;
		CUDA_CHECK( cudaDeviceGetAttribute( &display_dev, cudaDevAttrKernelExecTimeout, current_dev ) ) ;
		if ( display_dev>0 ) {
			SimpleUI simpleui( "RTWO", lp_general ) ;
			simpleui.usage() ;

			simpleui.render( pipeline, sbt ) ;
		} else {
			auto t0 = std::chrono::high_resolution_clock::now() ;
			// launch pipeline
			{
				CUstream cuda_stream ;
				CUDA_CHECK( cudaStreamCreate( &cuda_stream ) ) ;

				CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.image ), sizeof( uchar4 )*lp_general.image_w*lp_general.image_h ) ) ;

				CUdeviceptr d_lp_general ;
				const size_t lp_general_size = sizeof( LpGeneral ) ;
				CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_lp_general ), lp_general_size ) ) ;
				CUDA_CHECK( cudaMemcpy(
							reinterpret_cast<void*>( d_lp_general ),
							&lp_general,
							lp_general_size,
							cudaMemcpyHostToDevice
							) ) ;

				OPTX_CHECK( optixLaunch(
							pipeline,
							cuda_stream,
							d_lp_general,
							lp_general_size,
							&sbt,
							lp_general.image_w/*x*/, lp_general.image_h/*y*/, 1/*z*/ ) ) ;
				CUDA_CHECK( cudaDeviceSynchronize() ) ;

				CUDA_CHECK( cudaStreamDestroy( cuda_stream ) ) ;
				cudaError_t e = cudaGetLastError() ;
				if ( e != cudaSuccess ) {
					std::ostringstream comment ;
					comment << "CUDA error: " << cudaGetErrorString( e ) << "\n" ;
					throw std::runtime_error( comment.str() ) ;
				}
			}
			auto t1 = std::chrono::high_resolution_clock::now() ;
			long long int dt = std::chrono::duration_cast<std::chrono::milliseconds>( t1-t0 ).count() ;
			fprintf( stderr, "OptiX pipeline for RTWO ran %lld milliseconds\n", dt ) ;



			// output image
			std::vector<uchar4> image ;
			{
				const int w = lp_general.image_w ;
				const int h = lp_general.image_h ;
				image.resize( w*h ) ;
				CUDA_CHECK( cudaMemcpy(
							reinterpret_cast<void*>( image.data() ),
							lp_general.image,
							w*h*sizeof( uchar4 ),
							cudaMemcpyDeviceToHost
							) ) ;
				CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.image ) ) ) ;

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
		}



		// cleanup
		{
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_outbuf            ) ) ) ;
//			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_zipbuf            ) ) ) ;
			for ( unsigned int i = 0 ; things.size()>i ; i++ ) things[i] = nullptr ; // force thing's dtor

			OPTX_CHECK( optixPipelineDestroy    ( pipeline                ) ) ;
			OPTX_CHECK( optixProgramGroupDestroy( program_group_optics[OPTICS_TYPE_DIFFUSE] ) ) ;
			OPTX_CHECK( optixProgramGroupDestroy( program_group_optics[OPTICS_TYPE_REFLECT] ) ) ;
			OPTX_CHECK( optixProgramGroupDestroy( program_group_optics[OPTICS_TYPE_REFRACT] ) ) ;
			OPTX_CHECK( optixProgramGroupDestroy( program_group_ambient ) ) ;
			OPTX_CHECK( optixProgramGroupDestroy( program_group_camera  ) ) ;
			OPTX_CHECK( optixModuleDestroy      ( module_camera         ) ) ;
			OPTX_CHECK( optixModuleDestroy      ( module_optics         ) ) ;

			OPTX_CHECK( optixDeviceContextDestroy( optx_context ) ) ;
		}
	} catch ( std::exception& e ) {
		std::cerr << "exception: " << e.what() << "\n" ;

		return 1 ;
	}

	return 0 ;
}
