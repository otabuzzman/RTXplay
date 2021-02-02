#ifndef CAMERA_H
#define CAMERA_H

//#include <cmath>

#include <vector_functions.h>
#include <vector_types.h>

#include "util.h"
#include "v.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;

class Camera {
	public:
		Camera() ;

		void set( const float3&  eye, const float3&  pat, const float3&  vup, const float fov, const float aspratio, const float aperture, const float distance ) ;

		__device__ void ray( const float s, const float t, float3& ori, float3& dir ) const ;

	private:
		float3 u_, v_, w_ ;
		float3 eye_ ;
		float  lens_ ;
		float  dist_ ;
		float3 hvec_ ;
		float3 wvec_ ;
} ;

#endif // CAMERA_H
