#ifndef DENOISER_H
#define DENOISER_H

#include <vector_functions.h>
#include <vector_types.h>

class Denoiser {
	public:
		virtual ~Denoiser() noexcept ( false ) ;

		virtual void beauty( const float3* rawRGB, const float3* beauty = nullptr ) = 0 ;
} ;

class DenoiserSMP : public Denoiser {
	public:
		DenoiserSMP( const unsigned int w, const unsigned int h, const OptixDeviceContext optx_context ) ;
		~DenoiserSMP() noexcept ( false ) ;

		void beauty( const float3* rawRGB, const float3* beauty = nullptr ) noexcept ( false ) override ;

	private:
		unsigned int        w_ = 0 ;
		unsigned int        h_ = 0 ;
		OptixDenoiser       denoiser_ = {}  ;
		OptixDenoiserParams params_   = {}  ;
		CUdeviceptr         scratch_      = 0 ;
		unsigned int        scratch_size_ = 0 ;
		CUdeviceptr         state_        = 0 ;
		unsigned int        state_size_   = 0 ;
} ;

#endif // DENOISER_H
