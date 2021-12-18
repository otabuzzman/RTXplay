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
		unsigned int add( Thing& thing, unsigned int as_id ) ;   // create instance and connect with GAS

		bool set( unsigned int is_id, const float* transform ) ; // set instance transform
		bool get( unsigned int is_id, float* transform ) ;       // get instance transform

		void build( OptixTraversableHandle* is_handle ) ;
		void update( OptixTraversableHandle is_handle ) ;

	private:
		OptixDeviceContext optx_context_ ;

		// per object GAS handles
		std::vector<OptixTraversableHandle>   as_handle_ ;
		// per GAS device buffers
		std::vector<CUdeviceptr>              as_outbuf_ ;
		std::vector<std::vector<CUdeviceptr>> vces_ ;
		std::vector<std::vector<CUdeviceptr>> ices_ ;

		// per instance host data structures
		std::vector<OptixInstance>            is_ises_ ;
		// IAS device buffers
		CUdeviceptr                           is_outbuf_ ; // IAS output buffer
		CUdeviceptr                           is_updbuf_ ; // IAS updates buffer
		CUdeviceptr                           ises_ ;      // concatenated instances

		// per instance SBT record data buffers
		std::vector<Thing>                    things_ ;

		// data structures for IAS updates
		OptixBuildInput                       obi_ises_ ;
		size_t                                is_outbuf_size_ ;
		size_t                                is_updbuf_size_ ;

		void free() noexcept ( false ) ;
} ;

#endif // SCENE_H
