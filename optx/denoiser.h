#ifndef DENOISER_H
#define DENOISER_H

#include <vector>

#include <vector_functions.h>
#include <vector_types.h>

#include "args.h"

// common globals
extern OptixDeviceContext optx_context ;

class Denoiser {
	public:
		Denoiser( const Dns type, const unsigned int w, const unsigned int h ) ;
		~Denoiser() noexcept ( false ) ;

		void beauty( const float3* rawRGB, const float3* beauty = nullptr ) ;
		void guides( std::vector<float3>& normals, std::vector<float3>& albedos ) ;

	private:
		Dns                     type_ = Dns::NONE ;
		unsigned int            w_    = 0 ;
		unsigned int            h_    = 0 ;
		OptixDenoiser           denoiser_   = {} ;
		OptixDenoiserParams     params_     = {} ;
		OptixDenoiserGuideLayer guidelayer_ = {} ;
		CUdeviceptr             scratch_      = 0 ;
		unsigned int            scratch_size_ = 0 ;
		CUdeviceptr             state_        = 0 ;
		unsigned int            state_size_   = 0 ;
} ;

#endif // DENOISER_H
