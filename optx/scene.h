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

		Scene( const OptixDeviceContext& optx_context ) ;
		virtual ~Scene() noexcept ( false ) ;

		virtual unsigned int load() = 0 ;

		unsigned int add( Object& object ) ;                     // create GAS for object's submeshes
		unsigned int add( Thing& thing, unsigned int object ) ;  // create thing and connect with GAS

		bool set( unsigned int thing, const float* transform ) ; // set thing's transform
		bool set( unsigned int thing, unsigned int object ) ;    // set thing's object (GAS)

		void build( OptixTraversableHandle* is_handle ) ;
		void update() ;

	private:
		OptixDeviceContext optx_context_ ;

		// per object GAS handles
		std::vector<OptixTraversableHandle> as_handle_ ;
		// per GAS device buffers
		std::vector<CUdeviceptr>            as_outbuf_ ;
		std::vector<CUdeviceptr>            vces_ ;
		std::vector<CUdeviceptr>            ices_ ;

		// per instance host data structures
		std::vector<OptixInstance>          is_ises_ ;
		// IAS device buffers
		CUdeviceptr                         is_outbuf_ ; // IAS output buffer
		CUdeviceptr                         ises_ ;      // concatenated instances

		// per instance SBT record data buffers
		std::vector<Thing>                  things_ ;

		void free() noexcept ( false ) ;
} ;

#endif // SCENE_H
