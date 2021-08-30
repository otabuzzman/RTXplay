#include <optix.h>
#include <optix_stubs.h>

#include "util.h"

#include "denoiser.h"

Denoiser::Denoiser( const Dns type, const unsigned int w, const unsigned int h ) : type_( type ), w_( w ), h_( h ) {
	OptixDenoiserModelKind kind = type_ == Dns::AOV ? OPTIX_DENOISER_MODEL_KIND_AOV : OPTIX_DENOISER_MODEL_KIND_LDR ;

	OptixDenoiserOptions options ;
	if ( type_ == Dns::SMP )
		options = { 0, 0 } ;
	else if ( type_ == Dns::NRM ) {
		options = { 1, 0 } ;

		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &guidelayer_.normal.data ), w_*h_*sizeof( float3 ) ) ) ;
		guidelayer_.normal.width              = w_ ;
		guidelayer_.normal.height             = h_ ;
		guidelayer_.normal.rowStrideInBytes   = static_cast<unsigned int>( w_*sizeof( float3 ) ) ;
		guidelayer_.normal.pixelStrideInBytes = sizeof( float3 ) ;
		guidelayer_.normal.format             = OPTIX_PIXEL_FORMAT_FLOAT3 ;
	} else if ( type_ == Dns::ALB ) {
		options = { 0, 1 } ;

		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &guidelayer_.albedo.data ), w_*h_*sizeof( float3 ) ) ) ;
		guidelayer_.albedo.width              = w_ ;
		guidelayer_.albedo.height             = h_ ;
		guidelayer_.albedo.rowStrideInBytes   = static_cast<unsigned int>( w_*sizeof( float3 ) ) ;
		guidelayer_.albedo.pixelStrideInBytes = sizeof( float3 ) ;
		guidelayer_.albedo.format             = OPTIX_PIXEL_FORMAT_FLOAT3 ;
	} else { // if ( type_ == Dns::NAA || type_ == Dns::AOV )
		options = { 1, 1 } ;

		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &guidelayer_.normal.data ), w_*h_*sizeof( float3 ) ) ) ;
		guidelayer_.normal.width              = w_ ;
		guidelayer_.normal.height             = h_ ;
		guidelayer_.normal.rowStrideInBytes   = static_cast<unsigned int>( w_*sizeof( float3 ) ) ;
		guidelayer_.normal.pixelStrideInBytes = sizeof( float3 ) ;
		guidelayer_.normal.format             = OPTIX_PIXEL_FORMAT_FLOAT3 ;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &guidelayer_.albedo.data ), w_*h_*sizeof( float3 ) ) ) ;
		guidelayer_.albedo.width              = w_ ;
		guidelayer_.albedo.height             = h_ ;
		guidelayer_.albedo.rowStrideInBytes   = static_cast<unsigned int>( w_*sizeof( float3 ) ) ;
		guidelayer_.albedo.pixelStrideInBytes = sizeof( float3 ) ;
		guidelayer_.albedo.format             = OPTIX_PIXEL_FORMAT_FLOAT3 ;
	}

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
		) ) ;
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

	if ( guidelayer_.normal.data )
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( guidelayer_.normal.data ) ) ) ;
	if ( guidelayer_.albedo.data )
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( guidelayer_.albedo.data ) ) ) ;

	if ( type_ == Dns::AOV )
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params_.hdrIntensity ) ) ) ;
	else
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params_.hdrAverageColor ) ) ) ;
}

void Denoiser::beauty( const float3* rawRGB, const float3* beauty ) noexcept ( false ) {
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

	if ( type_ == Dns::AOV ) {
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &params_.hdrAverageColor ), 3*sizeof( float ) ) ) ;
		OPTX_CHECK( optixDenoiserComputeAverageColor(
			denoiser_,
			nullptr,
			&dns_layer.input,
			params_.hdrAverageColor,
			scratch_,
			scratch_size_
			) ) ;
	} else {
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &params_.hdrIntensity ), sizeof( float ) ) ) ;
		OPTX_CHECK( optixDenoiserComputeIntensity(
			denoiser_,
			nullptr,
			&dns_layer.input,
			params_.hdrIntensity,
			scratch_,
			scratch_size_
			) ) ;
	}

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

void Denoiser::guides( std::vector<float3>& normals, std::vector<float3>& albedos ) {
	if ( guidelayer_.normal.data ) {
		normals.resize( w_*h_ ) ;
		CUDA_CHECK( cudaMemcpy(
			normals.data(),
			reinterpret_cast<void*>( guidelayer_.normal.data ),
			w_*h_*sizeof( float3 ),
			cudaMemcpyDeviceToHost
			) ) ;
	}
	if ( guidelayer_.albedo.data ) {
		normals.resize( w_*h_ ) ;
		CUDA_CHECK( cudaMemcpy(
			albedos.data(),
			reinterpret_cast<void*>( guidelayer_.albedo.data ),
			w_*h_*sizeof( float3 ),
			cudaMemcpyDeviceToHost
			) ) ;
	}
}
