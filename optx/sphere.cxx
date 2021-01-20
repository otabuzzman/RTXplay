#include <vector>

#include <vector_functions.h>
#include <vector_types.h>

#include "v.h"

#include "thing.h"
#include "sphere.h"

using V::operator+ ;
using V::operator* ;

Sphere::Sphere( const float3& center, const float radius, const bool bbox )
	: center_( center ), radius_( radius ) {
	icosahedron( 4, bbox ) ;
}

void Sphere::icosahedron( const int ndiv, const bool bbox ) const {
	if ( bbox ) {
		// icosahedron's bounding box vertices
		vces_.add( { -radius,  radius, -radius } ) ;
		vces_.add( {  radius,  radius, -radius } ) ;
		vces_.add( {  radius, -radius, -radius } ) ;
		vces_.add( { -radius, -radius, -radius } ) ;
		vces_.add( { -radius,  radius,  radius } ) ;
		vces_.add( {  radius,  radius,  radius } ) ;
		vces_.add( {  radius, -radius,  radius } ) ;
		vces_.add( { -radius, -radius,  radius } ) ;

		// icosahedron's bounding box indices
		ices_.add( { 0, 1, 2 } ) ;
		ices_.add( { 2, 3, 0 } ) ;
		ices_.add( { 4, 5, 6 } ) ;
		ices_.add( { 6, 7, 4 } ) ;
		ices_.add( { 0, 1, 5 } ) ;
		ices_.add( { 5, 4, 0 } ) ;
		ices_.add( { 7, 6, 2 } ) ;
		ices_.add( { 2, 3, 7 } ) ;
		ices_.add( { 0, 4, 7 } ) ;
		ices_.add( { 7, 3, 0 } ) ;
		ices_.add( { 5, 1, 2 } ) ;
		ices_.add( { 2, 6, 5 } ) ;
	} else {
		// https://rechneronline.de/pi/icosahedron.php
		// r = a/4*sqrt(10+2*sqrt(5)) | r = 1
		// a = 1/sqrt(10+2*sqrt(5))*4
		// b = a/4*(1+sqrt(5))
		// precalculated values for unit icosahedron
		float a = 1.05146222424f/2.f ; // half edge length
		float b = 0.52573111211f     ; // midsphere radius

		// icosahedron's 12 vertices
		float3 v00 = { 0.f,   b,  -a } ;
		float3 v01 = {   b,   a, 0.f } ;
		float3 v02 = {  -b,   a, 0.f } ;
		float3 v03 = { 0.f,   b,   a } ;
		float3 v04 = { 0.f,  -b,   a } ;
		float3 v05 = {  -a, 0.f,   b } ;
		float3 v06 = { 0.f,  -b,  -a } ;
		float3 v07 = {   a, 0.f,  -b } ;
		float3 v08 = {   a, 0.f,   b } ;
		float3 v09 = {  -a, 0.f,  -b } ;
		float3 v10 = {   b,  -a, 0.f } ;
		float3 v11 = {  -b,  -a, 0.f } ;

		// icosahedron's 20 triangles
		divide( v00, v01, v02, ndiv ) ;
		divide( v03, v02, v01, ndiv ) ;
		divide( v03, v04, v05, ndiv ) ;
		divide( v03, v08, v04, ndiv ) ;
		divide( v00, v06, v07, ndiv ) ;
		divide( v00, v09, v06, ndiv ) ;
		divide( v04, v10, v11, ndiv ) ;
		divide( v06, v11, v10, ndiv ) ;
		divide( v02, v05, v09, ndiv ) ;
		divide( v11, v09, v05, ndiv ) ;
		divide( v01, v07, v08, ndiv ) ;
		divide( v10, v08, v07, ndiv ) ;
		divide( v03, v05, v02, ndiv ) ;
		divide( v03, v01, v08, ndiv ) ;
		divide( v00, v02, v09, ndiv ) ;
		divide( v00, v07, v01, ndiv ) ;
		divide( v06, v09, v11, ndiv ) ;
		divide( v06, v10, v07, ndiv ) ;
		divide( v04, v11, v05, ndiv ) ;
		divide( v04, v08, v10, ndiv ) ;
	}
}

void Sphere::divide( const float3& a, const float3& b, const float3& c, int ndiv ) const {
	if  ( ndiv>0 ) {
		float3 ab = V::unitV( .5f*( a+b ) ) ;
		float3 bc = V::unitV( .5f*( b+c ) ) ;
		float3 ca = V::unitV( .5f*( c+a ) ) ;

		divide (  a, ab, ca, ndiv-1 ) ;
		divide ( ab,  b, bc, ndiv-1 ) ;
		divide ( ca, bc,  c, ndiv-1 ) ;
		divide ( ab, bc, ca, ndiv-1 ) ;
	} else {
		vces_.add( center_+radius_*a ) ;
		vces_.add( center_+radius_*b ) ;
		vces_.add( center_+radius_*c ) ;
	}
}
