#ifndef THING_H
#define THING_H

#include <vector>

#include <vector_functions.h>
#include <vector_types.h>

#include "optics.h"

class Thing {
	public:
		__host__ __device__ float3*& d_vces() {
			return d_vces_ ;
		}

		unsigned int num_vces() {
			return static_cast<unsigned int>( vces_.size() ) ;
		}

		__host__ __device__ const uint3*  d_ices() const {
			return d_ices_ ;
		}

		unsigned int num_ices() {
			return static_cast<unsigned int>( ices_.size() ) ;
		}

		__device__ const float3 center() const {
			return center_ ;
		}

		__host__ __device__ const Optics optics() const {
			return optics_ ;
		}

	protected:
		float3 center_ ;
		Optics optics_ ;
		// CPU memory
		std::vector<float3> vces_ ; // thing's unique vertices...
		std::vector<uint3>  ices_ ; // ...as indexed triangles
		// GPU memory
		float3* d_vces_ ;
		uint3*  d_ices_ ;
} ;

#endif // THING_H
