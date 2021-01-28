//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <array>
#include <string>
#include <vector>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <sutil/Exception.h>

#include <vector_functions.h>
#include <vector_types.h>

#include "util.h"
#include "v.h"
#include "things.h"
#include "sphere.h"
#include "camera.h"

#include "optixTriangle.h"

using V::operator- ;
using V::operator* ;

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

extern "C" const unsigned char shader_optixTriangle[] ;

Things scene() {
	Things s ;

	s.push_back( make_shared<Sphere>( make_float3( 0.f, -1000.f, 0.f ), 1000.f ) ) ;

	for ( int a = -11 ; a<11 ; a++ ) {
		for ( int b = -11 ; b<11; b++ ) {
			auto select = util::rnd() ;
			float3 center = make_float3( a+.9f*util::rnd(), .2f, b+.9f*util::rnd() ) ;
			if ( V::len( center-make_float3( 4.f, .2f, 0.f ) )>.9f ) {
				if ( select<.8f ) {
					auto albedo = V::rnd()*V::rnd() ;
					s.push_back( make_shared<Sphere>( center, .2f ) ) ;
				} else if ( select<.95f ) {
					auto albedo = V::rnd( .5f, 1.f ) ;
					auto fuzz = util::rnd( 0.f, .5f ) ;
					s.push_back( make_shared<Sphere>( center, .2f ) ) ;
				} else {
					s.push_back( make_shared<Sphere>( center, .2f ) ) ;
				}
			}
		}
	}

	s.push_back( make_shared<Sphere>( make_float3(  0.f, 1.f, 0.f ), 1.f ) ) ;
	s.push_back( make_shared<Sphere>( make_float3( -4.f, 1.f, 0.f ), 1.f ) ) ;
	s.push_back( make_shared<Sphere>( make_float3(  4.f, 1.f, 0.f ), 1.f ) ) ;

	return s ;
}

int main() {
	Things things = scene() ;

	float aspratio = 3.f/2.f ;

	Camera camera(
		{13.f, 2.f, 3.f} /*eye*/,
		{ 0.f, 0.f, 0.f} /*pat*/,
		{ 0.f, 1.f, 0.f} /*vup*/,
		45.f /*fov*/,
		aspratio,
		.1f  /*aperture*/,
		10.f /*distance*/ ) ;

	const int w = 1200 ;                           // image width in pixels
	const int h = static_cast<int>( w/aspratio ) ; // image height in pixels



	try {
		// OptiX API log buffer and size varibale names must not change:
		// both hard-coded in OPTIX_CHECK_LOG macro (sutil/Exception.h).
		char   log[2048] ;
		size_t sizeof_log = sizeof( log ) ;



		// initialize
		OptixDeviceContext optx_context = nullptr;
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



		// build accelleration structure
		OptixTraversableHandle as_handle ;
		CUdeviceptr            d_as_outbuf ;
//		CUdeviceptr            d_as_zipbuf ;
		{
			std::vector<OptixBuildInput> obi_things ;
			obi_things.resize( things.size() ) ;

			std::vector<CUdeviceptr> d_vces ;
			d_vces.resize( things.size() ) ;

			std::vector<CUdeviceptr> d_ices ;
			d_ices.resize( things.size() ) ;

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
				const size_t ices_size = sizeof( float3 )*ices.size() ;
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
//				obi_thing.triangleArray.vertexStrideInBytes         = sizeof( float3 ) ;
				obi_thing.triangleArray.numVertices                 = static_cast<unsigned int>( vces.size() ) ;
				obi_thing.triangleArray.vertexBuffers               = &d_vces[i] ;

				obi_thing.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ;
//				obi_thing.triangleArray.indexStrideInBytes          = sizeof( uint3 ) ;
				obi_thing.triangleArray.numIndexTriplets            = static_cast<unsigned int>( ices.size() ) ;
				obi_thing.triangleArray.indexBuffer                 = d_ices[i] ;

				const unsigned int obiThingSbtFlags[1]              = { OPTIX_GEOMETRY_FLAG_NONE } ;
				obi_thing.triangleArray.flags                       = obiThingSbtFlags ;
				obi_thing.triangleArray.numSbtRecords               = 1 ;
				obi_thing.triangleArray.sbtIndexOffsetBuffer        = 0 ;
				obi_thing.triangleArray.sbtIndexOffsetSizeInBytes   = 0 ;
				obi_thing.triangleArray.sbtIndexOffsetStrideInBytes = 0 ;

				obi_things.push_back( obi_thing ) ;
			}

			OptixAccelBuildOptions oas_options = {} ;
			oas_options.buildFlags             = OPTIX_BUILD_FLAG_NONE ;
			oas_options.operation              = OPTIX_BUILD_OPERATION_BUILD ;

			OptixAccelBufferSizes as_buffer_sizes ;
			OPTIX_CHECK( optixAccelComputeMemoryUsage(
						optx_context,
						&oas_options,
						obi_things.data(),
						static_cast<unsigned int>( obi_things.size() ),
						&as_buffer_sizes
						) ) ;

			// optional AS buffer compaction I.
//			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_zipbuf_size ), sizeof( unsigned long long ) ) ) ;
//			CUDABuffer d_as_zipbuf_size ;

			// request optixAccelBuild to return compacted buffer size
//			OptixAccelEmitDesc oas_request;
//			oas_request.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE ;
//			oas_request.result = d_as_zipbuf_size ;

			CUdeviceptr d_as_tmpbuf ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_tmpbuf ), as_buffer_sizes.tempSizeInBytes ) ) ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_outbuf ), as_buffer_sizes.outputSizeInBytes ) ) ;

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
//						&oas_request, 1
						) ) ;

			// optional AS buffer compaction II.
//			unsigned long long as_zipbuf_size ;
//			CUDA_CHECK( cudaMemcpy(
//						static_cast<void*>( &as_zipbuf_size ),
//						d_as_zipbuf_size,
//						sizeof( unsigned long long ),
//						cudaMemcpyDeviceToHost
//						) ) ;

//			CUdeviceptr d_as_zipbuf ;
//			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_zipbuf ), as_zipbuf_size ) ) ;
//			OPTIX_CHECK(optixAccelCompact(
//						optixContext,
//						0,
//						as_handle,
//						d_as_zipbuf,
//						d_as_zipbuf_size,
//						&as_handle) ) ;

			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_tmpbuf ) ) ) ;
//			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_outbuf ) ) ) ;
//			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_zipbuf_size ) ) ) ;
			for ( const CUdeviceptr p : d_vces ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( p ) ) ) ;
			for ( const CUdeviceptr p : d_ices ) CUDA_CHECK( cudaFree( reinterpret_cast<void*>( p ) ) ) ;
		}

        //
        // Create module
        //
        OptixModule optx_module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

            pipeline_compile_options.usesMotionBlur        = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues      = 3;
            pipeline_compile_options.numAttributeValues    = 3;
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "lpGeneral";
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            const std::string ptx( reinterpret_cast<char const*>( shader_optixTriangle ) );
            OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                        optx_context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        ptx.c_str(),
                        ptx.size(),
                        log,
                        &sizeof_log,
                        &optx_module
                        ) );
        }

        //
        // Create program groups
        //
        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = optx_module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        optx_context,
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &raygen_prog_group
                        ) );

            OptixProgramGroupDesc miss_prog_group_desc  = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = optx_module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        optx_context,
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &miss_prog_group
                        ) );

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH            = optx_module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        optx_context,
                        &hitgroup_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &hitgroup_prog_group
                        ) );
        }

        //
        // Link optx_pipeline
        //
        OptixPipeline optx_pipeline = nullptr;
        {
            const uint32_t    max_trace_depth  = 1;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth          = max_trace_depth;
            pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            OPTIX_CHECK_LOG( optixPipelineCreate(
                        optx_context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups,
                        sizeof( program_groups ) / sizeof( program_groups[0] ),
                        log,
                        &sizeof_log,
                        &optx_pipeline
                        ) );

            OptixStackSizes stack_sizes = {};
            for( auto& prog_group : program_groups )
            {
                OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                     0,  // maxCCDepth
                                                     0,  // maxDCDEpth
                                                     &direct_callable_stack_size_from_traversal,
                                                     &direct_callable_stack_size_from_state, &continuation_stack_size ) );
            OPTIX_CHECK( optixPipelineSetStackSize( optx_pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state, continuation_stack_size,
                                                    1  // maxTraversableDepth
                                                    ) );
        }

        //
        // Set up shader binding table
        //
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            RayGenSbtRecord rg_sbt;
            memcpy( &rg_sbt.data, &camera, sizeof( Camera ) ) ;
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( raygen_record ),
                        &rg_sbt,
                        raygen_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            ms_sbt.data = { 0.3f, 0.1f, 0.2f };
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( miss_record ),
                        &ms_sbt,
                        miss_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr hitgroup_record;
            size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
            HitGroupSbtRecord hg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( hitgroup_record ),
                        &hg_sbt,
                        hitgroup_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
            sbt.hitgroupRecordCount         = 1;
        }



		// launch
		uchar4* d_image ;
		CUDA_CHECK( cudaMalloc(
					reinterpret_cast<void**>( &d_image ),
					w*h*sizeof( uchar4 )
					) ) ;

		{
			CUstream cuda_stream ;
			CUDA_CHECK( cudaStreamCreate( &cuda_stream ) ) ;

			LpGeneral lpGeneral ; // launch parameter

			lpGeneral.image       = d_image ;
			lpGeneral.image_w     = w ;
			lpGeneral.image_h     = h ;

			lpGeneral.as_handle  = as_handle ;

			CUdeviceptr d_lpGeneral ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_lpGeneral ), sizeof( LpGeneral ) ) ) ;
			CUDA_CHECK( cudaMemcpy(
						reinterpret_cast<void*>( d_lpGeneral ),
						&lpGeneral, sizeof( lpGeneral ),
						cudaMemcpyHostToDevice
						) ) ;

			OPTIX_CHECK( optixLaunch( optx_pipeline, cuda_stream, d_lpGeneral, sizeof( LpGeneral ), &sbt, w, h, /*depth=*/1 ) ) ;
			CUDA_SYNC_CHECK() ;
		}



		// output image
		{
			std::vector<uchar4>  h_image ;
			h_image.resize( w*h ) ;
			CUDA_CHECK( cudaMemcpy(
						static_cast<void*>( h_image.data() ),
						d_image,
						w*h*sizeof( uchar4 ),
						cudaMemcpyDeviceToHost
						) ) ;

			std::cout
				<< "P3\n" // magic PPM header
				<< w << ' ' << h << '\n' << 255 << '\n' ;

			for ( int y = h-1 ; y>=0 ; --y ) {
				for ( int x = 0 ; x<w ; ++x ) {
					auto p = h_image.data()[w*y+x] ;
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
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_as_outbuf           ) ) ) ;
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_image                ) ) ) ;

			OPTIX_CHECK( optixPipelineDestroy    ( optx_pipeline       ) ) ;
			OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) ) ;
			OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group     ) ) ;
			OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group   ) ) ;
			OPTIX_CHECK( optixModuleDestroy      ( optx_module         ) ) ;

			OPTIX_CHECK( optixDeviceContextDestroy( optx_context ) ) ;
		}
	} catch( std::exception& e ) {
		std::cerr << "exception: " << e.what() << "\n" ;

		return 1 ;
	}

	return 0 ;
}
