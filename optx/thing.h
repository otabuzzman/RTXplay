#ifndef THING_H
#define THING_H

#include <vector>

#include <vector_functions.h>
#include <vector_types.h>

#include "optics.h"

class Thing {
	public:
		virtual ~Thing() noexcept ( false ) {} ;

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

		virtual void transform( float const matrix[12] ) {
			for ( int i = 0 ; 12>i ; i++ )
				matrix_[i] = matrix[i] ;
		}

		__host__ __device__ const float* transform() const {
			return &matrix_[0] ;
		}

		__host__ __device__ const Optics optics() const {
			return optics_ ;
		}

	protected:
		// row-major 3x4, pre-multiplication
		float  matrix_[12] = {
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
			0, 0, 0
		} ;
		Optics optics_ ;
		// CPU memory
		std::vector<float3> vces_ ; // thing's unique vertices...
		std::vector<uint3>  ices_ ; // ...as indexed triangles
		// GPU memory
		float3* d_vces_ ;
		uint3*  d_ices_ ;
} ;

#endif // THING_H
