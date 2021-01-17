#include <vector_functions.h>
#include <vector_types.h>

#include "v.h"
#include "util.h"

#include "camera.h"

using V::operator- ;
using V::operator* ;

Camera::Camera( const float3&  eye, const float3&  pat, const float3&  vup, const float fov, const float aspratio )
	: eye_( eye ), pat_( pat ), vup_( vup ), fov_( fov ), aspratio_( aspratio ) {}

void Camera::set( LpCamera& camera ) const {
	camera.eye  = eye_ ;
	float3 W    = pat_ - eye_; // Do not normalize W -- it implies focal length
	float wlen  = V::len( W );
	float3 U    = V::unitV( V::cross( W, vup_ ) ) ;
	float3 V    = V::unitV( V::cross( U, W ) ) ;
	
	camera.w    = W ;
	float vlen  = wlen*tanf( .5f*fov_*util::kPi / 180.f ) ;
	camera.v    = V*vlen ;
	float ulen  = vlen*aspratio_ ;
	camera.u    = U*ulen ;
}
