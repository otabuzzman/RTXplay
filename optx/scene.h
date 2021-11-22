#ifndef SCENE_H
#define SCENE_H

// system includes
#include <vector>

// subsystem includes
// OptiX
#include <optix.h>
#include <optix_stubs.h>

// local includes
#include "object.h"
#include "thing.h"

// file specific includes
// none

class Scene {
	public:
		const Thing& operator[] ( unsigned int i ) { return things_[i] ; } ;

		Scene() = default ;
		Scene( const OptixDeviceContext& optx_context ) ;
		virtual ~Scene() noexcept ( false ) ;

		virtual void load( unsigned int* size ) = 0 ;

		unsigned int add( Object& object ) ;                                            // create GAS for object's submeshes
		unsigned int add( Thing& thing, const float* transform, unsigned int object ) ; // create thing and connect with GAS

		void build( OptixTraversableHandle* handle ) ;

	private:
		OptixDeviceContext optx_context_ ;

		// per object GAS handles
		std::vector<OptixTraversableHandle> as_handle_ ;
		// per GAS device buffers
		std::vector<CUdeviceptr>            as_outbuf_ ;
		std::vector<CUdeviceptr>            vces_ ;
		std::vector<CUdeviceptr>            ices_ ;

		// per instance host data structures
		std::vector<OptixInstance> h_ises_ ;
		// per instance SBT record data buffers
		std::vector<Thing>         things_ ;

		// IAS device buffer
		CUdeviceptr is_outbuf_ ;
		// catenated instances device buffer
		CUdeviceptr d_ises_ ;
} ;

#endif // SCENE_H
