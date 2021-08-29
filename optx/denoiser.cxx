#include <optix.h>
#include <optix_stubs.h>

#include "util.h"

#include "denoiser.h"

DenoiserNRM::DenoiserNRM( const unsigned int w, const unsigned int h ) : Denoiser( w, h, OPTIX_DENOISER_MODEL_KIND_LDR, OptixDenoiserOptions{ 1, 0 } ) {}

DenoiserNRM::~DenoiserNRM() noexcept ( false ) {
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params_.hdrIntensity ) ) ) ;
}

void DenoiserNRM::beauty( const float3* rawRGB, const float3* beauty ) noexcept ( false ) {
}

DenoiserALB::DenoiserALB( const unsigned int w, const unsigned int h ) : Denoiser( w, h, OPTIX_DENOISER_MODEL_KIND_LDR, OptixDenoiserOptions{ 0, 1 } ) {}

DenoiserALB::~DenoiserALB() noexcept ( false ) {
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params_.hdrIntensity ) ) ) ;
}

void DenoiserALB::beauty( const float3* rawRGB, const float3* beauty ) noexcept ( false ) {
}

DenoiserNAA::DenoiserNAA( const unsigned int w, const unsigned int h ) : Denoiser( w, h, OPTIX_DENOISER_MODEL_KIND_LDR, OptixDenoiserOptions{ 1, 1 } ) {}

DenoiserNAA::~DenoiserNAA() noexcept ( false ) {
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params_.hdrIntensity ) ) ) ;
}

void DenoiserNAA::beauty( const float3* rawRGB, const float3* beauty ) noexcept ( false ) {
}

DenoiserAOV::DenoiserAOV( const unsigned int w, const unsigned int h ) : Denoiser( w, h, OPTIX_DENOISER_MODEL_KIND_AOV, OptixDenoiserOptions{ 1, 1 } ) {}

DenoiserAOV::~DenoiserAOV() noexcept ( false ) {
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params_.hdrAverageColor ) ) ) ;
}

void DenoiserAOV::beauty( const float3* rawRGB, const float3* beauty ) noexcept ( false ) {
}

DenoiserSMP::DenoiserSMP( const unsigned int w, const unsigned int h ) : Denoiser( w, h, OPTIX_DENOISER_MODEL_KIND_LDR, OptixDenoiserOptions{ 0, 0 } ) {}

DenoiserSMP::~DenoiserSMP() noexcept ( false ) {
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

	OPTX_CHECK( optixDenoiserInvoke(
		denoiser_,
		nullptr,
		&params_,
		state_,
		state_size_,
		&guidelayer_,
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

Denoiser::Denoiser( const unsigned int w, const unsigned int h, const OptixDenoiserModelKind kind, const OptixDenoiserOptions options ) : w_( w ), h_( h ) {
	OPTX_CHECK( optixDenoiserCreate(
		optx_context,
		kind,
		&options,
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

Denoiser::~Denoiser() noexcept ( false ) {
	OPTX_CHECK( optixDenoiserDestroy( denoiser_ ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( scratch_   ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state_     ) ) ) ;
}
