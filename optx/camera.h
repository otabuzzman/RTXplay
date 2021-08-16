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
		void set( const float3&  eye, const float3&  pat, const float3&  vup, const float fov, const float aspratio, const float aperture, const float distance ) {
			w_    = V::unitV( eye-pat ) ;
			u_    = V::unitV( V::cross( vup, w_ ) ) ;
			v_    = V::cross( w_, u_ ) ;

			float hlen  = 2.f*tanf( .5f*fov*util::kPi/180.f ) ;
			float wlen  = hlen*aspratio ;

			eye_  = eye ;
			lens_ = aperture/2.f ;
			dist_ = distance ;
			hvec_ = hlen*v_ ;
			wvec_ = wlen*u_ ;
		} ;

#ifdef __CUDACC__

		__device__ void ray( const float s, const float t, float3& ori, float3& dir, curandState* state ) const {
			const float3 r = lens_*V::rndVin1disk( state ) ;
			const float3 o = r.x*u_+r.y*v_ ;

			ori = eye_+o ;
			dir = dist_*( s*wvec_+t*hvec_-w_ )-o ;
		} ;

#endif

	private:
		float3 u_, v_, w_ ;
		float3 eye_ ;
		float  lens_ ;
		float  dist_ ;
		float3 hvec_ ;
		float3 wvec_ ;
} ;

#endif // CAMERA_H
