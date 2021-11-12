#ifndef DENOISER_H
#define DENOISER_H

// system includes
// none

// subsystem includes
// OptiX
#include <optix.h>
// CUDA
#include <vector_types.h>

// local includes
#include "args.h"

// file specific includes
// none

class Denoiser {
	public:
		Denoiser( const Dns type ) ;
		~Denoiser() noexcept ( false ) ;

		void beauty( const float3* rawRGB, const float3* beauty = nullptr ) ;
		const Dns type() const { return type_ ; } ;

	private:
		Dns                     type_ = Dns::NONE ;
		OptixDenoiser           denoiser_   = {} ;
		OptixDenoiserParams     params_     = {} ;
		OptixDenoiserGuideLayer guidelayer_ = {} ;
		CUdeviceptr             scratch_      = 0 ;
		unsigned int            scratch_size_ = 0 ;
		CUdeviceptr             state_        = 0 ;
		unsigned int            state_size_   = 0 ;
} ;

#endif // DENOISER_H
