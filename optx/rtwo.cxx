// system includes
#include <array>
#include <chrono>
#include <cstring>
#include <vector>

// subsystem includes
// OptiX
#include <optix.h>
#include <optix_stubs.h>
/*** calculate stack sizes
***/
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>
// CUDA
#include <vector_functions.h>
#include <vector_types.h>
// GLFW
#include <GLFW/glfw3.h>

// local includes
#include "args.h"
#include "camera.h"
#include "denoiser.h"
#include "hoist.h"
#include "scene.h"
#include "simpleui.h"
#include "sphere.h"
#include "util.h"
#include "v.h"

// file specific includes
#include "rtwo.h"

// common globals
Args*              args ;
LpGeneral          lp_general ;
OptixDeviceContext optx_context ;

// PTX sources of shaders
extern "C" const char camera_r_ptx[] ; // recursive
extern "C" const char optics_r_ptx[] ;
extern "C" const char camera_i_ptx[] ; // iterative
extern "C" const char optics_i_ptx[] ;

#ifdef RECURSIVE
const char* camera_ptx = &camera_r_ptx[0] ;
const char* optics_ptx = &optics_r_ptx[0] ;
#else
const char* camera_ptx = &camera_i_ptx[0] ;
const char* optics_ptx = &optics_i_ptx[0] ;
#endif //RECURSIVE

// post processing
extern "C" void pp_none( const float3* src, uchar4* dst, const int w, const int h ) ;
extern "C" void pp_sRGB( const float3* src, uchar4* dst, const int w, const int h ) ;

// output local image on stdout
static void imgtopnm( const std::vector<uchar4>       img, const int w, const int h ) ; // output PPM, ignore A channel
static void imgtopnm( const std::vector<float3>       img, const int w, const int h ) ; // output PPM
static void imgtopnm( const std::vector<unsigned int> img, const int w, const int h ) ; // output PGM
// output device image on stdout
template<typename T>
static void imgtopnm( const CUdeviceptr img, const int w, const int h ) {
	std::vector<T> image ;
	image.resize( w*h ) ;
	CUDA_CHECK( cudaMemcpy(
		image.data(),
		reinterpret_cast<void*>( img ),
		w*h*sizeof( T ),
		cudaMemcpyDeviceToHost
		) ) ;
	imgtopnm( image, w, h ) ;
}

// build geometry-AS from number of hoists
static void makeGAS( const Hoist* scene, size_t scene_size, OptixTraversableHandle* is_handle, CUdeviceptr* d_as_outbuf, CUdeviceptr* d_as_zipbuf ) ;



int main( int argc, char* argv[] ) {
	args = new Args( argc, argv ) ;

	if ( args->flag_h() ) {
		Args::usage() ;
		SimpleUI::usage() ;

		return 0 ;
	}

	lp_general.image_w = args->param_w( 1280 ) ;          // image width in pixels
	lp_general.image_h = args->param_h( 720 )  ;          // image height in pixels
	lp_general.spp     = args->param_s( 50 ) ;            // samples per pixel
#ifdef RECURSIVE
#define MAX_DEPTH 16
#else
#define MAX_DEPTH 50
#endif // RECURSIVE
	const int depth    = args->param_d( MAX_DEPTH ) ; // recursion depth or number of iterations
	lp_general.depth   = depth>MAX_DEPTH ? MAX_DEPTH : depth ;

	float aspratio = static_cast<float>( lp_general.image_w )/static_cast<float>( lp_general.image_h ) ;
	lp_general.camera.set(
		{ 13.f, 2.f, 3.f } /*eye*/,
		{  0.f, 0.f, 0.f } /*pat*/,
		{  0.f, 1.f, 0.f } /*vup*/,
		20.f /*fov*/,
		aspratio,
		.1f  /*aperture*/,
		10.f /*focus distance*/ ) ;

	SbtRecordMS sbt_record_ambient ;
	sbt_record_ambient.data = { .5f, .7f, 1.f } ;

	try {
		// initialize
		{
			CUDA_CHECK( cudaFree( 0 ) ) ;
			OPTX_CHECK( optixInit() ) ;

			OptixDeviceContextOptions optx_options = {} ;
			optx_options.logCallbackFunction       = &util::optxLogStderr ;
			optx_options.logCallbackLevel          = args->flag_v() ? 4 : 0 ;
			optx_options.validationMode            = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL ;

			// use current (0) CUDA context
			CUcontext cuda_context = 0 ;
			OPTX_CHECK( optixDeviceContextCreate( cuda_context, &optx_options, &optx_context ) ) ;
		}



		// build acceleration structure
		CUdeviceptr d_is_outbuf ;                       // IAS
		std::vector<OptixTraversableHandle> as_handle ; // one GAS per UTM
		std::vector<CUdeviceptr> d_as_outbuf ;
		std::vector<CUdeviceptr> d_as_zipbuf ;
		Scene scene ;
		{
			size_t num_utm ;
			scene.load( &num_utm ) ;
			as_handle.resize( num_utm, 0 ) ;
			d_as_outbuf.resize( num_utm, 0 ) ;
			d_as_zipbuf.resize( num_utm, 0 ) ;

			// instance structures of things in scene
			const size_t scene_size = scene.size() ;
			std::vector<OptixInstance> ises ;
			ises.resize( scene_size ) ;

			// create instance structure for each thing in scene
			const size_t transform_size = sizeof( float )*12 ;
			for ( unsigned int i = 0 ; scene_size>i ; i++ ) {
				OptixInstance instance     = {} ;

				instance.instanceId        = i ;
				instance.visibilityMask    = 255 ; // OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK

				instance.flags             = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;
				instance.sbtOffset         = i ;

				memcpy( instance.transform, scene[i].transform, transform_size ) ;

				const int utm_index = scene[i].utm_index ;
				if ( ! as_handle[utm_index] )
					makeGAS( &scene[i], 1, &as_handle[utm_index], &d_as_outbuf[utm_index], &d_as_zipbuf[utm_index] ) ;
				instance.traversableHandle = as_handle[scene[i].utm_index] ;

				ises[i] = instance ;
			}

			CUdeviceptr d_ises ;
			const size_t instances_size = sizeof( OptixInstance )*ises.size() ;
			CUDA_CHECK( cudaMalloc( (void**) &d_ises, instances_size) );
			CUDA_CHECK( cudaMemcpy( (void*) d_ises, ises.data(), instances_size, cudaMemcpyHostToDevice ) );

			// create build input structure for thing instances
			OptixBuildInput obi_things            = {} ;
			obi_things.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			obi_things.instanceArray.instances    = d_ises ;
			obi_things.instanceArray.numInstances = static_cast<unsigned int>( ises.size() ) ;

			OptixAccelBuildOptions ois_options    = {} ;
			ois_options.buildFlags                = OPTIX_BUILD_FLAG_NONE ;
			ois_options.operation                 = OPTIX_BUILD_OPERATION_BUILD ;

			// request acceleration structure buffer sizes from OptiX
			OptixAccelBufferSizes is_buffer_sizes ;
			OPTX_CHECK( optixAccelComputeMemoryUsage(
						optx_context,
						&ois_options,
						&obi_things,
						1,
						&is_buffer_sizes
						) ) ;

			// allocate GPU memory for acceleration structure buffers
			CUdeviceptr d_is_tmpbuf ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_is_tmpbuf ), is_buffer_sizes.tempSizeInBytes ) ) ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_is_outbuf ), is_buffer_sizes.outputSizeInBytes ) ) ;

			OPTX_CHECK( optixAccelBuild(
						optx_context,
						0,
						&ois_options,
						&obi_things,
						1,
						d_is_tmpbuf,
						is_buffer_sizes.tempSizeInBytes,
						d_is_outbuf,
						is_buffer_sizes.outputSizeInBytes,
						&lp_general.is_handle,
						nullptr, 0
						) ) ;

			CUDA_CHECK( cudaFree( (void*) d_ises      ) ) ;
			CUDA_CHECK( cudaFree( (void*) d_is_tmpbuf ) ) ;
		}



		// create module(s)
		OptixModule module_camera = nullptr ;
		OptixModule module_optics = nullptr ;
		OptixPipelineCompileOptions pipeline_cc_options = {} ;
		{
			OptixModuleCompileOptions module_cc_options = {} ;
			module_cc_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT ;
			module_cc_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT ; // OPTIX_COMPILE_OPTIMIZATION_LEVEL_0
			module_cc_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT ;  // OPTIX_COMPILE_DEBUG_LEVEL_FULL

			pipeline_cc_options.usesMotionBlur                   = false ;
			pipeline_cc_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING ;
#ifdef RECURSIVE
			pipeline_cc_options.numPayloadValues                 = 8 ; // R, G, B, RNG (2x), depth, DGV (2x)
#else
			pipeline_cc_options.numPayloadValues                 = 2 ; // RayParam (2x)
#endif // RECURSIVE
			pipeline_cc_options.numAttributeValues               = 2 ;
			pipeline_cc_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE ;
			pipeline_cc_options.pipelineLaunchParamsVariableName = "lp_general" ;
			pipeline_cc_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ;

			// compile (each) shader source file into a module
			const size_t camera_ptx_size = strlen( camera_ptx ) ;
			OPTX_CHECK_LOG( optixModuleCreateFromPTX(
						optx_context,
						&module_cc_options,
						&pipeline_cc_options,
						camera_ptx,
						camera_ptx_size,
						log,
						&sizeof_log,
						&module_camera
						) ) ;

			const size_t optics_ptx_size = strlen( optics_ptx ) ;
			OPTX_CHECK_LOG( optixModuleCreateFromPTX(
						optx_context,
						&module_cc_options,
						&pipeline_cc_options,
						optics_ptx,
						optics_ptx_size,
						log,
						&sizeof_log,
						&module_optics
						) ) ;
		}



		// create program groups
		OptixProgramGroup program_group_camera    = nullptr ;
		OptixProgramGroup program_group_ambient   = nullptr ;
		OptixProgramGroup program_group_optics[Optics::TYPE_NUM] = {} ;
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
			program_group_optics_desc[Optics::TYPE_DIFFUSE].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP ;
			program_group_optics_desc[Optics::TYPE_DIFFUSE].hitgroup.entryFunctionNameCH = "__closesthit__diffuse" ;
			program_group_optics_desc[Optics::TYPE_DIFFUSE].hitgroup.moduleCH            = module_optics ;
			program_group_optics_desc[Optics::TYPE_REFLECT].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP ;
			program_group_optics_desc[Optics::TYPE_REFLECT].hitgroup.moduleCH            = module_optics ;
			program_group_optics_desc[Optics::TYPE_REFLECT].hitgroup.entryFunctionNameCH = "__closesthit__reflect" ;
			program_group_optics_desc[Optics::TYPE_REFRACT].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP ;
			program_group_optics_desc[Optics::TYPE_REFRACT].hitgroup.moduleCH            = module_optics ;
			program_group_optics_desc[Optics::TYPE_REFRACT].hitgroup.entryFunctionNameCH = "__closesthit__refract" ;
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
#ifdef RECURSIVE
			const unsigned int max_trace_depth  = lp_general.depth ;
#else
			const unsigned int max_trace_depth  = 1 ;
#endif // RECURSIVE
			OptixProgramGroup program_groups[]  = {
				program_group_camera,
				program_group_ambient,
				program_group_optics[Optics::TYPE_DIFFUSE],
				program_group_optics[Optics::TYPE_REFLECT],
				program_group_optics[Optics::TYPE_REFRACT] } ;

			OptixPipelineLinkOptions pipeline_ld_options = {} ;
			pipeline_ld_options.maxTraceDepth            = max_trace_depth ;
			pipeline_ld_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT ;  // OPTIX_COMPILE_DEBUG_LEVEL_FULL
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



			/*** calculate stack sizes (see `fixed stack sizes´ alternative below)
			***/
			OptixStackSizes stack_sizes = {} ;
			for ( auto& prog_group : program_groups )
				OPTX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) ) ;
//			fprintf( stderr, "cssRG: %u, cssMS: %u, cssCH: %u, cssAH: %u, cssIS: %u, cssCC: %u, dssDC: %u\n",
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

			OPTX_CHECK( optixPipelineSetStackSize(
						pipeline,
						dssTrav, // direct callable stack size (called from AH and IS programs)
						dssStat, // direct callable stack size (called from RG, MS and CH programs)
						css,     // continuation callable stack size
						2        // maxTraversableGraphDepth (acceleration structure depth)
						) ) ;



			/*** fixed stack sizes (see `calculate stack sizes´ alternative above)
			OPTX_CHECK( optixPipelineSetStackSize(
						pipeline,
						8*1024, // direct callable stack size (called from AH and IS programs)
						8*1024, // direct callable stack size (called from RG, MS and CH programs)
						8*1024, // continuation callable stack size
						2       // maxTraversableGraphDepth (acceleration structure depth)
						) ) ;
			***/
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
			sbt_record_buffer.resize( scene.size() ) ;

			// set SBT record for each thing in scene
			for ( unsigned int i = 0 ; scene.size()>i ; i++ ) {
				// this thing's SBT record
				SbtRecordHG sbt_record_thing ;
				sbt_record_thing.data = scene[i] ;
				// setup SBT record header
				OPTX_CHECK( optixSbtRecordPackHeader( program_group_optics[scene[i].optics.type], &sbt_record_thing ) ) ;
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
			SimpleUI simpleui( "RTWO" ) ;
			simpleui.render( pipeline, sbt ) ;
		} else {
			const int w = lp_general.image_w ;
			const int h = lp_general.image_h ;
			std::vector<unsigned int> rpp ;

			{ // launch pipeline
				CUstream cuda_stream ;
				CUDA_CHECK( cudaStreamCreate( &cuda_stream ) ) ;

				if ( args->param_D( Dns::NONE ) != Dns::NONE ) {
					lp_general.spp = 1 ;
					if ( args->param_D( Dns::NONE ) == Dns::NRM )
						CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.normals ), sizeof( float3 )*w*h ) ) ;
					else if ( args->param_D( Dns::NONE ) == Dns::ALB )
						CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.albedos ), sizeof( float3 )*w*h ) ) ;
					else { // if ( args->param_D( Dns::NONE ) == Dns::NAA || args->param_D( Dns::NONE ) == Dns::AOV )
						CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.normals ), sizeof( float3 )*w*h ) ) ;
						CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.albedos ), sizeof( float3 )*w*h ) ) ;
					}
				}
				CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.rawRGB ), sizeof( float3 )*w*h ) ) ;
				CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.rpp ), sizeof( unsigned int )*w*h ) ) ;

				CUdeviceptr d_lp_general ;
				const size_t lp_general_size = sizeof( LpGeneral ) ;
				CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_lp_general ), lp_general_size ) ) ;
				CUDA_CHECK( cudaMemcpy(
							reinterpret_cast<void*>( d_lp_general ),
							&lp_general,
							lp_general_size,
							cudaMemcpyHostToDevice
							) ) ;

				auto t0 = std::chrono::high_resolution_clock::now() ;
				OPTX_CHECK( optixLaunch(
							pipeline,
							cuda_stream,
							d_lp_general,
							lp_general_size,
							&sbt,
							w/*x*/, h/*y*/, 1/*z*/ ) ) ;
				CUDA_CHECK( cudaDeviceSynchronize() ) ;
				auto t1 = std::chrono::high_resolution_clock::now() ;

				CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_lp_general ) ) ) ;

				if ( args->flag_S() ) { // output statistics
					long long dt = std::chrono::duration_cast<std::chrono::milliseconds>( t1-t0 ).count() ;
					rpp.resize( w*h ) ;
					CUDA_CHECK( cudaMemcpy(
								rpp.data(),
								lp_general.rpp,
								w*h*sizeof( unsigned int ),
								cudaMemcpyDeviceToHost
								) ) ;
					long long sr = 0 ; for ( auto const& c : rpp ) sr = sr+c ; // accumulate rays per pixel
					fprintf( stderr, "%9u %12llu %4llu (pixel, rays, milliseconds) %6.2f fps\n", w*h, sr, dt, 1000.f/dt ) ;
				}

				CUDA_CHECK( cudaStreamDestroy( cuda_stream ) ) ;
				if ( cudaGetLastError() != cudaSuccess ) {
					std::ostringstream comment ;
					comment << "CUDA error: " << cudaGetErrorString( cudaGetLastError() ) << "\n" ;
					throw std::runtime_error( comment.str() ) ;
				}
			}



			// apply denoiser
			const Dns type = args->param_D( Dns::NONE ) ;
			if ( type != Dns::NONE ) {
				Denoiser* denoiser = new Denoiser( type, w, h ) ;
				denoiser->beauty( lp_general.rawRGB ) ;

				// output guides
				if ( ! args->flag_q() && args->flag_G() ) {
					if ( lp_general.normals ) imgtopnm<float3>( reinterpret_cast<CUdeviceptr>( lp_general.normals ), w, h ) ;
					if ( lp_general.albedos ) imgtopnm<float3>( reinterpret_cast<CUdeviceptr>( lp_general.albedos ), w, h ) ;
				}

				delete denoiser ;
			}



			// post processing
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.image ), sizeof( uchar4 )*w*h ) ) ;
			pp_sRGB( lp_general.rawRGB, lp_general.image, w, h ) ;



			// output image
			if ( ! args->flag_q() )
				imgtopnm<uchar4>( reinterpret_cast<CUdeviceptr>( lp_general.image ), w, h ) ;

			// output AOV rays per pixel (RPP)
			if ( ! args->flag_q() && args->flag_A( Aov::RPP ) ) {
				if ( rpp.size() == 0 )
					imgtopnm<unsigned int>( reinterpret_cast<CUdeviceptr>( lp_general.rpp ), w, h ) ;
				else
					imgtopnm( rpp, w, h ) ;
			}

			if ( lp_general.normals ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.normals    ) ) ) ;
			if ( lp_general.albedos ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.albedos    ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.rawRGB ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.image  ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.rpp    ) ) ) ;
		}



		// cleanup
		{
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_is_outbuf            ) ) ) ;
			for ( auto b : d_as_outbuf ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;
//			for ( auto b : d_as_zipbuf ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( b ) ) ) ;

			OPTX_CHECK( optixPipelineDestroy    ( pipeline              ) ) ;
			OPTX_CHECK( optixProgramGroupDestroy( program_group_optics[Optics::TYPE_DIFFUSE] ) ) ;
			OPTX_CHECK( optixProgramGroupDestroy( program_group_optics[Optics::TYPE_REFLECT] ) ) ;
			OPTX_CHECK( optixProgramGroupDestroy( program_group_optics[Optics::TYPE_REFRACT] ) ) ;
			OPTX_CHECK( optixProgramGroupDestroy( program_group_ambient ) ) ;
			OPTX_CHECK( optixProgramGroupDestroy( program_group_camera  ) ) ;
			OPTX_CHECK( optixModuleDestroy      ( module_camera         ) ) ;
			OPTX_CHECK( optixModuleDestroy      ( module_optics         ) ) ;

			OPTX_CHECK( optixDeviceContextDestroy( optx_context ) ) ;

			delete args ;
		}
	} catch ( const std::exception& e ) {
		std::cerr << "exception: " << e.what() << "\n" ;

		return 1 ;
	}

	return 0 ;
}

static void imgtopnm( const std::vector<uchar4> rgb, const int w, const int h ) {
	std::cout
		<< "P3\n" // magic PPM header
		<< w << ' ' << h << '\n' << 255 << '\n' ;

	for ( int y = h-1 ; y>=0 ; --y ) {
		for ( int x = 0 ; x<w ; ++x ) {
			auto p = rgb.data()[w*y+x] ;
			std::cout
				<< static_cast<int>( p.x ) << ' '
				<< static_cast<int>( p.y ) << ' '
				<< static_cast<int>( p.z ) << '\n' ;
		}
	}
	std::cout << std::endl ;
}

static void imgtopnm( const std::vector<float3> rgb, const int w, const int h ) {
	std::cout
		<< "P3\n" // magic PPM header
		<< w << ' ' << h << '\n' << 255 << '\n' ;

	for ( int y = h-1 ; y>=0 ; --y ) {
		for ( int x = 0 ; x<w ; ++x ) {
			auto p = rgb.data()[w*y+x] ;
			std::cout
				<< static_cast<int>( util::clamp( p.x, 0.f, 1.f )*255 ) << ' '
				<< static_cast<int>( util::clamp( p.y, 0.f, 1.f )*255 ) << ' '
				<< static_cast<int>( util::clamp( p.z, 0.f, 1.f )*255 ) << '\n' ;
		}
	}
	std::cout << std::endl ;
}

static void imgtopnm( const std::vector<unsigned int> mono, const int w, const int h ) {
	std::cout
		<< "P2\n" // magic PGM header
		<< w << ' ' << h << '\n' << 65535 << '\n' ;

	for ( int y = h-1 ; y>=0 ; --y )
		for ( int x = 0 ; x<w ; ++x )
			std::cout
				<< mono.data()[w*y+x] << '\n' ;
	std::cout << std::endl ;
}

static void makeGAS( const Hoist* scene, size_t scene_size, OptixTraversableHandle* is_handle, CUdeviceptr* d_as_outbuf, CUdeviceptr* d_as_zipbuf ) {
	// build input structures of things in scene
	std::vector<OptixBuildInput> obi_things ;
	obi_things.resize( scene_size ) ;

	// GPU pointers at vertices lists of things in scene
	std::vector<CUdeviceptr> vces ;
	vces.resize( scene_size ) ;

	// GPU pointers at indexed triangles lists of things
	std::vector<CUdeviceptr> ices ;
	ices.resize( scene_size ) ;

	// thing specific (or one fits all) flags per SBT record
	// std::vector<std::vector<unsigned int>> obi_thing_flags ;
	// obi_thing_flags.resize( scene_size ) ;
	const unsigned int obi_thing_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT } ;

	// create build input structure for each thing in scene
	for ( unsigned int i = 0 ; scene_size>i ; i++ ) {
		vces[i] = reinterpret_cast<CUdeviceptr>( scene[i].vces ) ;
		ices[i] = reinterpret_cast<CUdeviceptr>( scene[i].ices ) ;
		// setup this thing's build input structure
		OptixBuildInput obi_thing = {} ;
		obi_thing.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES ;

		obi_thing.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3 ;
		obi_thing.triangleArray.numVertices                 = scene[i].num_vces ;
		obi_thing.triangleArray.vertexBuffers               = &vces[i] ;

		obi_thing.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ;
		obi_thing.triangleArray.numIndexTriplets            = scene[i].num_ices ;
		obi_thing.triangleArray.indexBuffer                 = ices[i] ;

		// obi_thing_flags[i].push_back( OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ) ;
		// obi_thing.triangleArray.flags                       = &obi_thing_flags[i][0] ;
		obi_thing.triangleArray.flags                       = &obi_thing_flags[0] ;
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
//	CUDABuffer d_as_zipbuf_size ;
//	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_zipbuf_size ), sizeof( unsigned long long ) ) ) ;

	// provide request description for acceleration structure compaction buffer size
//	OptixAccelEmitDesc oas_request ;
//	oas_request.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE ;
//	oas_request.result = d_as_zipbuf_size ;

	// allocate GPU memory for acceleration structure buffers
	CUdeviceptr d_as_tmpbuf ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_tmpbuf ), as_buffer_sizes.tempSizeInBytes ) ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(  d_as_outbuf ), as_buffer_sizes.outputSizeInBytes ) ) ;

	// build acceleration structure
	OPTX_CHECK( optixAccelBuild(
				optx_context,
				0,
				&oas_options,
				obi_things.data(),
				static_cast<unsigned int>( obi_things.size() ),
				d_as_tmpbuf,
				as_buffer_sizes.tempSizeInBytes,
				*d_as_outbuf,
				as_buffer_sizes.outputSizeInBytes,
				is_handle,
				nullptr, 0
				// acceleration structure compaction (must comment previous line)
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
//	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_zipbuf ), as_zipbuf_size ) ) ;
	// condense previously built acceleration structure
//	OPTX_CHECK( optixAccelCompact(
//				optixContext,
//				0,
//				lp_general.is_handle,
//				d_as_zipbuf,
//				d_as_zipbuf_size,
//				&lp_general.is_handle ) ) ;

	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_tmpbuf ) ) ) ;
	// free GPU memory for acceleration structure compaction buffer
//	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_zipbuf_size ) ) ) ;
}
