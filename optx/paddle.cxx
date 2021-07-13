#include <algorithm>
#include <cmath>

#include "util.h"
#include "v.h"

#include "paddle.h"

using V::operator+ ;
using V::operator- ;
using V::operator* ;

Paddle::Paddle( const float3& eye, const float3& pat, const float3& vup ) {
	const float3 d = V::unitV( eye-pat ) ;
	const float x = V::dot( d, u_ ) ;
	const float y = V::dot( d, v_ ) ;
	const float z = V::dot( d, w_ ) ;
	lo_ = atan2( x, y ) ;
	la_ = asin ( z ) ;
}

void Paddle::start( const int x, const int y ) {
	x_ = x ;
	y_ = y ;
}

float3 Paddle::move( const int x, const int y ) {
	const int dx = x-x_ ;
	const int dy = y-y_ ;
	x_ = x ;
	y_ = y ;
	lo_ = util::rad( fmod( util::deg( lo_ )-.25f*dx, 360.f ) ) ;
	la_ = util::rad( std::min( 89.999f, std::max( -89.999f, util::deg( la_ )+.25f*dy ) ) ) ;

	const float u = cos( la_ )*sin( lo_ ) ;
	const float v = cos( la_ )*cos( lo_ ) ;
	const float w = sin( la_ ) ;

	return u*u_+v*v_+w*w_ ;
}
