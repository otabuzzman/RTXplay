#include <optix.h>
#include <optix_stubs.h>

#include "util.h"

#include "denoiser.h"

DenoiserSMP::DenoiserSMP( const unsigned int w, const unsigned int h, const OptixDeviceContext optx_context ) : w_( w ), h_( h ) {
	OptixDenoiserOptions dns_options = {} ;
	OPTX_CHECK( optixDenoiserCreate(
		optx_context,
		OPTIX_DENOISER_MODEL_KIND_LDR ,
		&dns_options,
		&denoiser_
		) ) ;

	OptixDenoiserSizes dns_sizes ;
	OPTX_CHECK( optixDenoiserComputeMemoryResources(
		denoiser_,
		w_,
		h_,
		&dns_sizes
		) );
	scratch_size_ = static_cast<unsigned int>( dns_sizes.withoutOverlapScratchSizeInBytes ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &scratch_ ), scratch_size_ ) ) ;
	state_size_ = static_cast<unsigned int>( dns_sizes.stateSizeInBytes );
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state_ ), state_size_ ) ) ;

	OPTX_CHECK( optixDenoiserSetup(
		denoiser_,
		nullptr,
		w_,
		h_,
		state_,
		state_size_,
		scratch_,
		scratch_size_
		) ) ;
}

DenoiserSMP::~DenoiserSMP() noexcept ( false ) {
	OPTX_CHECK( optixDenoiserDestroy( denoiser_ ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( scratch_   ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state_     ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params_.hdrIntensity ) ) ) ;
}

void DenoiserSMP::beauty( const float3* rawRGB, const float3* beauty ) noexcept ( false ) {
	OptixDenoiserLayer dns_layer = {} ;
	dns_layer.input = {
		reinterpret_cast<CUdeviceptr>( rawRGB ),
		w_,
		h_,
		static_cast<unsigned int>( w_*sizeof( float3 ) ),
		sizeof( float3 ),
		OPTIX_PIXEL_FORMAT_FLOAT3
		} ;
	dns_layer.output = {
		beauty ? reinterpret_cast<CUdeviceptr>( beauty ) : reinterpret_cast<CUdeviceptr>( rawRGB ),
		w_,
		h_,
		static_cast<unsigned int>( w_*sizeof( float3 ) ),
		sizeof( float3 ),
		OPTIX_PIXEL_FORMAT_FLOAT3
		} ;

	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &params_.hdrIntensity ), sizeof( float ) ) ) ;
	OPTX_CHECK( optixDenoiserComputeIntensity(
		denoiser_,
		nullptr,
		&dns_layer.input,
		params_.hdrIntensity,
		scratch_,
		scratch_size_
		) ) ;

	OptixDenoiserGuideLayer dns_guidelayer = {} ;
	OPTX_CHECK( optixDenoiserInvoke(
		denoiser_,
		nullptr,
		&params_,
		state_,
		state_size_,
		&dns_guidelayer,
		&dns_layer,
		1,
		0,
		0,
		scratch_,
		scratch_size_
		) ) ;

	CUDA_CHECK( cudaDeviceSynchronize() ) ;
	if ( cudaGetLastError() != cudaSuccess ) {
		std::ostringstream comment ;
		comment << "CUDA error: " << cudaGetErrorString( cudaGetLastError() ) << "\n" ;
		throw std::runtime_error( comment.str() ) ;
	}
}
