#ifndef DENOISER_H
#define DENOISER_H

#include <vector_functions.h>
#include <vector_types.h>

class DenoiserSMP {
	public:
		DenoiserSMP( const unsigned int w, const unsigned int h, const OptixDeviceContext optx_context ) ;
		~DenoiserSMP() noexcept ( false ) ;

		void update( const float3* rawRGB ) ;
		float3* beauty() ;

	private:
		unsigned int        w_ ;
		unsigned int        h_ ;
		OptixDenoiser       denoiser_     ;
		OptixDenoiserLayer  layer_  = {}  ;
		OptixDenoiserParams params_ = {}  ;
		CUdeviceptr         scratch_      ;
		unsigned int        scratch_size_ ;
		CUdeviceptr         state_        ;
		unsigned int        state_size_   ;
		float3*             beauty_       ;
} ;

#endif // DENOISER_H
