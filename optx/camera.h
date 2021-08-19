#ifndef CAMERA_H
#define CAMERA_H

#include <vector_functions.h>
#include <vector_types.h>

#include "util.h"
#include "v.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;

class Camera {
	public:

#ifndef __CUDACC__

		void set( const float3& eye, const float3& pat, const float3&  vup, const float fov, const float aspratio, const float aperture, const float fostance ) {
			eye_ = eye ;
			pat_ = pat ;
			vup_ = vup ;
			fov_ = fov ;
			aspratio_ = aspratio ;
			aperture_ = aperture ;
			fostance_ = fostance ; // focus distance

			w_    = V::unitV( eye_-pat_ ) ;
			u_    = V::unitV( V::cross( vup_, w_ ) ) ;
			v_    = V::cross( w_, u_ ) ;

			float h  = 2.f*tanf( .5f*util::rad( fov_ ) ) ; // focus plane (virtual viewport) height
			float w  = h*aspratio_ ;                       // focus plane width
			hvec_ = fostance_*h/2.f*v_ ; // focus plane height vector
			wvec_ = fostance_*w/2.f*u_ ; // focus plane width vector
			dvec_ = fostance_      *w_ ; // lens/ focus plane distance vector (focus distance)
		} ;

		float3 eye()                      const { return eye_ ; }
		void   eye( const float3& eye )         { set( eye,  pat_, vup_, fov_, aspratio_, aperture_, fostance_ ) ; }
		float3 pat()                      const { return pat_ ; }
		void   pat( const float3& pat )         { set( eye_, pat,  vup_, fov_, aspratio_, aperture_, fostance_ ) ; }
		float3 vup()                      const { return vup_ ; }
		void   vup( const float3& vup )         { set( eye_, pat_, vup,  fov_, aspratio_, aperture_, fostance_ ) ; }
		float  fov()                      const { return fov_ ; }
		void   fov( const float fov )           { set( eye_, pat_, vup_, fov,  aspratio_, aperture_, fostance_ ) ; }
		void   aspratio( const float aspratio ) { set( eye_, pat_, vup_, fov_, aspratio,  aperture_, fostance_ ) ; }
		float  aperture()                 const { return aperture_ ; }
		void   aperture( const float aperture ) { set( eye_, pat_, vup_, fov_, aspratio_, aperture,  fostance_ ) ; }

#else

#ifdef CURAND
		__device__ void ray( const float s, const float t, float3& ori, float3& dir, curandState* state ) const {
#else
		__device__ void ray( const float s, const float t, float3& ori, float3& dir, Frand48* state ) const {
#endif // CURAND
			// defocus blur (depth of field)
			const float3 r = aperture_/2.f*V::rndVin1disk( state ) ;
			const float3 o = r.x*u_+r.y*v_ ;

			ori = eye_+o ;
			dir = s*wvec_+t*hvec_-dvec_-o ;
		} ;

#endif

	private:
		float3 eye_ ;
		float3 pat_ ;
		float3 vup_ ;
		float  fov_ ;
		float  aspratio_ ;
		float  aperture_ ;
		float  fostance_ ;

		float3 u_, v_, w_ ;
		float3 hvec_ ;
		float3 wvec_ ;
		float3 dvec_ ;
} ;

#endif // CAMERA_H
