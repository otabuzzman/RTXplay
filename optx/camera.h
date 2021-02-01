#ifndef CAMERA_H
#define CAMERA_H

#include <cmath>

#include <vector_functions.h>
#include <vector_types.h>

#include "util.h"
#include "v.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;

class Camera {
	public:
		__host__ Camera( const float3&  eye, const float3&  pat, const float3&  vup, const float fov, const float aspratio, const float aperture, const float distance ) {
			w_    = V::unitV( eye-pat ) ;
			u_    = V::unitV( V::cross( w_, vup ) ) ;
			v_    = V::cross( u_, w_ ) ;

			float hlen  = 2.f*tanf( .5f*fov*util::kPi/180.f ) ;
			float wlen  = hlen*aspratio ;

			eye_  = eye ;
			lens_ = aperture/2.f ;
			dist_ = distance ;
			hvec_ = hlen*v_ ;
			wvec_ = wlen*u_ ;
		}

		__device__ void ray( const float s, const float t, float3& ori, float3& dir ) const {
			ori = eye_ ;
			dir = dist_*( s*wvec_+t*hvec_ )-eye_ ;
		}

	private:
		float3 u_ ;
		float3 v_ ;
		float3 w_ ;
		float3 eye_ ;
		float  lens_ ;
		float  dist_ ;
		float3 hvec_ ;
		float3 wvec_ ;
} ;

#endif // CAMERA_H
