#include <vector_functions.h>
#include <vector_types.h>

#include "v.h"
#include "util.h"

#include "camera.h"

using V::operator- ;
using V::operator* ;

Camera::Camera( const float3&  eye, const float3&  pat, const float3&  vup, const float fov, const float aspratio, const float aperture, const float distance )
	: eye_( eye ), pat_( pat ), vup_( vup ), fov_( fov ), aspratio_( aspratio ), aperture_( aperture ), distance_( distance ) {}

void Camera::set( LpCamera& camera ) const {
	camera.w    = V::unitV( eye_-pat_ ) ;
	camera.u    = V::unitV( V::cross( camera.w, vup_ ) ) ;
	camera.v    = V::cross( camera.u, camera.w ) ;

	float hlen  = 2.f*tanf( .5f*fov_*util::kPi/180.f ) ;
	float wlen  = hlen*aspratio_ ;

	camera.eye  = eye_ ;
	camera.lens = aperture_/2.f ;
	camera.dist = distance_ ;
	camera.hvec = hlen*camera.v ;
	camera.wvec = wlen*camera.u ;
}
