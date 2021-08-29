#ifndef DENOISER_H
#define DENOISER_H

#include <vector_functions.h>
#include <vector_types.h>

// common globals
extern OptixDeviceContext optx_context ;

class Denoiser {
	public:
		Denoiser( const unsigned int w, const unsigned int h, const OptixDenoiserModelKind kind, const OptixDenoiserOptions options ) ;
		virtual ~Denoiser() noexcept ( false ) ;

		virtual void beauty( const float3* rawRGB, const float3* beauty = nullptr ) = 0 ;

	protected:
		unsigned int        w_ = 0 ;
		unsigned int        h_ = 0 ;
		OptixDenoiser       denoiser_ = {}  ;
		OptixDenoiserParams params_   = {}  ;
		CUdeviceptr         scratch_      = 0 ;
		unsigned int        scratch_size_ = 0 ;
		CUdeviceptr         state_        = 0 ;
		unsigned int        state_size_   = 0 ;
} ;

class DenoiserSMP : public Denoiser {
	public:
		DenoiserSMP( const unsigned int w, const unsigned int h ) ;
		~DenoiserSMP() noexcept ( false ) ;

		void beauty( const float3* rawRGB, const float3* beauty = nullptr ) noexcept ( false ) override ;
} ;

class DenoiserNRM : public Denoiser {
	public:
		DenoiserNRM( const unsigned int w, const unsigned int h ) ;
		~DenoiserNRM() noexcept ( false ) ;

		void beauty( const float3* rawRGB, const float3* beauty = nullptr ) noexcept ( false ) override ;
} ;

class DenoiserALB : public Denoiser {
	public:
		DenoiserALB( const unsigned int w, const unsigned int h ) ;
		~DenoiserALB() noexcept ( false ) ;

		void beauty( const float3* rawRGB, const float3* beauty = nullptr ) noexcept ( false ) override ;
} ;

class DenoiserNAA : public Denoiser {
	public:
		DenoiserNAA( const unsigned int w, const unsigned int h ) ;
		~DenoiserNAA() noexcept ( false ) ;

		void beauty( const float3* rawRGB, const float3* beauty = nullptr ) noexcept ( false ) override ;
} ;

class DenoiserAOV : public Denoiser {
	public:
		DenoiserAOV( const unsigned int w, const unsigned int h ) ;
		~DenoiserAOV() noexcept ( false ) ;

		void beauty( const float3* rawRGB, const float3* beauty = nullptr ) noexcept ( false ) override ;
} ;

#endif // DENOISER_H
