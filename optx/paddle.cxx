#include <algorithm>
#include <cmath>

#include "util.h"
#include "v.h"

#include "paddle.h"

using V::operator+ ;
using V::operator* ;

Paddle::Paddle( const float3& hand ) {
	const float3 h = V::unitV( hand ) ;
	const float x = V::dot( h, u_ ) ;
	const float y = V::dot( h, v_ ) ;
	const float z = V::dot( h, w_ ) ;
	lo_ = atan2( x, y ) ;
	la_ = asin ( z ) ;
}

void Paddle::start( const int x, const int y ) {
	x_ = x ;
	y_ = y ;
}

void Paddle::track( const int x, const int y ) {
	const int dx = x-x_ ;
	const int dy = y-y_ ;
	x_ = x ;
	y_ = y ;
	lo_ = util::rad( fmod( util::deg( lo_ )-.25f*dx, 360.f ) ) ;
	la_ = util::rad( std::min( 89.999f, std::max( -89.999f, util::deg( la_ )+.25f*dy ) ) ) ;
}

float3 Paddle::hand() const {
	const float x = cos( la_ )*sin( lo_ ) ;
	const float y = cos( la_ )*cos( lo_ ) ;
	const float z = sin( la_ ) ;

	return x*u_+y*v_+z*w_ ;
}
