// system includes
#include <algorithm>

// subsystem includes
// none

// local includes
#include "util.h"
#include "v.h"

// file specific includes
#include "paddle.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;

Paddle::Paddle( const float3& eye, const float3& pat, const float3& vup ) {
	gauge( eye, pat, vup ) ;
}

void Paddle::gauge( const float3& eye, const float3& pat, const float3& vup ) {
	// initialize move
	const float3 d = V::unitV( eye-pat ) ;
	float x = V::dot( d, u_ ) ;
	float y = V::dot( d, v_ ) ;
	float z = V::dot( d, w_ ) ;
	lo_ = atan2f( x, y ) ;
	la_ = asinf ( z ) ;

	// initialize roll
	vup_ = V::unitV( vup ) ;
	x = V::dot( vup_, u_ ) ;
	y = V::dot( vup_, w_ ) ;
	phi_ = atan2f( y, x ) ;
}

void Paddle::reset( const int x, const int y ) {
	x_ = x ;
	y_ = y ;
}

float3 Paddle::move( const int x, const int y, int* deltax, int* deltay, const float stepping ) {
	const int dx = x-x_ ;
	const int dy = y-y_ ;
	x_ = x ;
	y_ = y ;
	lo_ = util::rad( fmod( util::deg( lo_ )-stepping*float(dx), 360.f ) ) ;
	la_ = util::rad( std::min( 89.999f, std::max( -89.999f, util::deg( la_ )+stepping*float(dy) ) ) ) ;

	const float u = cosf( la_ )*sinf( lo_ ) ;
	const float v = cosf( la_ )*cosf( lo_ ) ;
	const float w = sinf( la_ ) ;

	if ( deltax ) *deltax = dx ;
	if ( deltay ) *deltay = dy ;

	return u*u_+v*v_+w*w_ ;
}

float3 Paddle::roll( const int s, const float stepping ) {
	phi_ = util::rad( fmod( util::deg( phi_ )+stepping*float(s), 360.f ) ) ;
	const float x = cosf( phi_ ) ;
	const float y = sinf( phi_ ) ;
	const float z = vup_.z ;

	return make_float3( x, y, z ) ;
}
