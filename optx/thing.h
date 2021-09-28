#ifndef THING_H
#define THING_H

#include <cuda.h>

#include <vector>

#include <vector_functions.h>
#include <vector_types.h>

#include "optics.h"

class Thing {
	public:
		Thing() {
			const size_t matrix_size = sizeof( float )*12 ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &matrix_ ), matrix_size ) ) ;
		}
		~Thing() noexcept ( false ) {
			CUDA_CHECK( cudaFree( reinterpret_cast<void*>( matrix_ ) ) ) ;
		}

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

		void transform( float const matrix[12] ) {
			const size_t matrix_size = sizeof( float )*12 ;
			CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( matrix_ ),
				matrix,
				matrix_size,
				cudaMemcpyHostToDevice
				) ) ;
		}

		__host__ __device__ const CUdeviceptr transform() const {
			return matrix_ ;
		}

		__host__ __device__ const Optics optics() const {
			return optics_ ;
		}

	private:
		// row-major 3x4 matrix (pre-multiplication 1x3 vector)
		CUdeviceptr matrix_ ;

	protected:
		Optics optics_ ;
		// CPU memory
		std::vector<float3> vces_ ; // thing's unique vertices...
		std::vector<uint3>  ices_ ; // ...as indexed triangles
		// GPU memory
		float3* d_vces_ ;
		uint3*  d_ices_ ;
} ;

#endif // THING_H
