#ifndef LAUNCHER_H
#define LAUNCHER_H

// system includes
// none

// subsystem includes
// OptiX
#include <optix.h>

// local includes
// none

// file specific includes
// none

class Launcher {
	public:
		Launcher( const OptixPipeline& pipeline, const OptixShaderBindingTable& sbt ) ;
		~Launcher() noexcept ( false ) ;

		void resize( const unsigned int w, const unsigned int h ) ;
		void ignite( const CUstream& cuda_stream, bool once = false ) ;

	private:
		CUdeviceptr             lp_general_ ;
		OptixPipeline           pipeline_   ;
		OptixShaderBindingTable sbt_        ;

		void free() noexcept ( false ) ;
} ;

#endif // LAUNCHER_H
