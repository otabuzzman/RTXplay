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
		void set( const float3& eye, const float3& pat, const float3&  vup, const float fov, const float aspratio, const float aperture, const float distance ) {
			eye_ = eye ;
			pat_ = pat ;
			vup_ = vup ;
			fov_ = fov ;
			aspratio_ = aspratio ;
			aperture_ = aperture ;
			distance_ = distance ;

			w_    = V::unitV( eye_-pat_ ) ;
			u_    = V::unitV( V::cross( vup_, w_ ) ) ;
			v_    = V::cross( w_, u_ ) ;

			float hlen  = 2.f*tanf( .5f*util::rad( fov_ ) ) ;
			float wlen  = hlen*aspratio_ ;

			lens_ = aperture_/2.f ;
			hvec_ = hlen*v_ ;
			wvec_ = wlen*u_ ;
		} ;

		float3 eye()              const { return eye_ ; }
		void   eye( const float3& eye ) { set( eye, pat_, vup_, fov_, aspratio_, aperture_, distance_ ) ; }
		float3 pat()              const { return pat_ ; }
		void   pat( const float3& pat ) { set( eye_, pat, vup_, fov_, aspratio_, aperture_, distance_ ) ; }

#ifdef __CUDACC__

		__device__ void ray( const float s, const float t, float3& ori, float3& dir, curandState* state ) const {
			const float3 r = lens_*V::rndVin1disk( state ) ;
			const float3 o = eye_+r.x*u_+r.y*v_ ;

			ori = o ;
			dir = distance_*( s*wvec_+t*hvec_ )-o ;
		} ;

#endif

	private:
		float3 eye_ ;
		float3 pat_ ;
		float3 vup_ ;
		float  fov_ ;
		float  aspratio_ ;
		float  aperture_ ;
		float  distance_ ;

		float3 u_, v_, w_ ;
		float  lens_ ;
		float3 hvec_ ;
		float3 wvec_ ;
} ;

#endif // CAMERA_H
