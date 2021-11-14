// system includes
#include <chrono>

// subsystem includes
// OptiX
#include <optix.h>
#include <optix_stubs.h>
// CUDA
#include <vector_functions.h>
#include <vector_types.h>

// local includes
#include "args.h"
#include "rtwo.h"

// file specific includes
#include "launcher.h"

// common globals
extern Args*                   args ;
extern LpGeneral               lp_general ;

Launcher::Launcher( const OptixPipeline& pipeline, const OptixShaderBindingTable& sbt ) : pipeline_( pipeline ), sbt_( sbt ) {
	// allocate device memory for launch parameters
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general_ ), sizeof( LpGeneral ) ) ) ;
	// various buffers
	resize( lp_general.image_w, lp_general.image_h ) ;
}

Launcher::~Launcher() noexcept ( false ) {
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general_        ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.rawRGB  ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.image   ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.rpp     ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.normals ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.albedos ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.pick_id ) ) ) ;
}

void Launcher::resize( const unsigned int w, const unsigned int h ) {
	// render result buffer
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.rawRGB ),  sizeof( float3 )*w*h ) ) ;
	// post processing buffer
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.image ),   sizeof( uchar4 )*w*h ) ) ;
	// AOV rays per pixel (RPP) buffer
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.rpp ),     sizeof( unsigned int )*w*h ) ) ;
	// denoiser input buffers
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.normals ), sizeof( float3 )*w*h ) ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.albedos ), sizeof( float3 )*w*h ) ) ;
	// picked instance id buffer (scene edit)
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.pick_id ), sizeof( unsigned int ) ) ) ;
}

void Launcher::ignite( const CUstream& cuda_stream, const unsigned int w, const unsigned int h ) {
	const unsigned int w0 = w ? w : lp_general.image_w ;
	const unsigned int h0 = h ? h : lp_general.image_h ;

	// update launch parameters
	const size_t lp_general_size = sizeof( LpGeneral ) ;
	CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( lp_general_ ),
				&lp_general,
				lp_general_size,
				cudaMemcpyHostToDevice
				) ) ;

	// launch pipeline
	auto t0 = std::chrono::high_resolution_clock::now() ;
	OPTX_CHECK( optixLaunch(
				pipeline_,
				cuda_stream,
				lp_general_,
				lp_general_size,
				&sbt_,
				w0/*x*/, h0/*y*/, 1/*z*/ ) ) ;
	CUDA_CHECK( cudaDeviceSynchronize() ) ;
	auto t1 = std::chrono::high_resolution_clock::now() ;

	if ( cudaGetLastError() != cudaSuccess ) {
		std::ostringstream comment ;
		comment << "CUDA error: " << cudaGetErrorString( cudaGetLastError() ) << "\n" ;
		throw std::runtime_error( comment.str() ) ;
	}

	// output statistics
	if ( args->flag_S() ) {
		std::vector<unsigned int> rpp ;
		rpp.resize( w0*h0 ) ;
		CUDA_CHECK( cudaMemcpy(
					rpp.data(),
					lp_general.rpp,
					w0*h0*sizeof( unsigned int ),
					cudaMemcpyDeviceToHost
					) ) ;
		long long dt = std::chrono::duration_cast<std::chrono::milliseconds>( t1-t0 ).count() ;
		long long sr = 0 ; for ( auto const& c : rpp ) sr = sr+c ; // accumulate rays per pixel
		fprintf( stderr, "%9u %12llu %4llu (pixel, rays, milliseconds) %6.2f fps\n", w0*h0, sr, dt, 1000.f/dt ) ;
	}
}
