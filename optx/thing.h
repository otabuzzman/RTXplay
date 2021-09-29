#ifndef THING_H
#define THING_H

#include <cuda.h>

#include <vector>

#include <vector_functions.h>
#include <vector_types.h>

#include "optics.h"
#include "util.h"

class Thing {
	public:
		__host__ __device__ const float3* d_vces() const {
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

#ifndef __CUDACC__

		void transform( float const matrix[12] ) {
			CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( matrix_ ),
				matrix,
				sizeof( float )*12,
				cudaMemcpyHostToDevice
				) ) ;
		}

#endif // __CUDACC__

		__host__ __device__ const float* transform() const {
			return matrix_ ;
		}

		__host__ __device__ const Optics optics() const {
			return optics_ ;
		}

	protected:
		// row-major 3x4 matrix
		float* matrix_ ;
		Optics optics_ ;
		// CPU memory
		std::vector<float3> vces_ ; // thing's unique vertices...
		std::vector<uint3>  ices_ ; // ...as indexed triangles
		// GPU memory
		float3* d_vces_ ;
		uint3*  d_ices_ ;
} ;

#endif // THING_H
