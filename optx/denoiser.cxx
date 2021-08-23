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
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( beauty_    ) ) ) ;
}

void DenoiserSMP::update( const float3* rawRGB ) {
	layer_.input = {
		reinterpret_cast<CUdeviceptr>( rawRGB ),
		w_,
		h_,
		w_*sizeof( float3 ),
		sizeof( float3 ),
		OPTIX_PIXEL_FORMAT_FLOAT3
		} ;
}

float3* DenoiserSMP::beauty() {
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &beauty_ ), w_*h_*sizeof( float3 ) ) ) ;
	layer_.output = {
		reinterpret_cast<CUdeviceptr>( beauty_ ),
		w_,
		h_,
		w_*sizeof( float3 ),
		sizeof( float3 ),
		OPTIX_PIXEL_FORMAT_FLOAT3
		} ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &params_.hdrIntensity ), sizeof( float ) ) ) ;
	OPTX_CHECK( optixDenoiserComputeIntensity(
		denoiser_,
		nullptr,
		&layer_.input,
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
		&layer_,
		1,
		0,
		0,
		scratch_,
		scratch_size_
		) ) ;
	CUDA_CHECK( cudaDeviceSynchronize() ) ;

	return beauty_ ;
}
