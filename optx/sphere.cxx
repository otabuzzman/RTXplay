#include <set>
#include <vector>

#include <vector_functions.h>
#include <vector_types.h>

#include "v.h"

#include "thing.h"
#include "sphere.h"

using V::operator+ ;
using V::operator* ;

Sphere::Sphere( const float3& center, const float radius, const bool bbox, const uint ndiv )
	: center_( center ), radius_( radius ), ndiv_( ndiv ) {
	tetrahedron( bbox ) ;
}

void Sphere::tetrahedron( const bool bbox ) {
	vces_.clear() ;
	ices_.clear() ;

	// https://rechneronline.de/pi/tetrahedron.php
	// r = a/4*sqrt(6) | r = 1
	// a = 4/sqrt(6)
	// m = a/4*sqrt(2)
	// precalculated value for unit tetrahedron
	float m = .57735026919f ; // midsphere radius

	if ( bbox ) {
		// tetrahedron's bounding box vertices
		float3 v00 = { -m,  m, -m } ;
		float3 v01 = {  m,  m, -m } ;
		float3 v02 = {  m, -m, -m } ;
		float3 v03 = { -m, -m, -m } ;
		float3 v04 = { -m,  m,  m } ;
		float3 v05 = {  m,  m,  m } ;
		float3 v06 = {  m, -m,  m } ;
		float3 v07 = { -m, -m,  m } ;

		// tetrahedron's bounding box triangles
		vtmp_.push_back( v00 ) ; vtmp_.push_back( v01 ) ; vtmp_.push_back( v02 ) ;
		vtmp_.push_back( v02 ) ; vtmp_.push_back( v03 ) ; vtmp_.push_back( v00 ) ;
		vtmp_.push_back( v04 ) ; vtmp_.push_back( v05 ) ; vtmp_.push_back( v06 ) ;
		vtmp_.push_back( v06 ) ; vtmp_.push_back( v07 ) ; vtmp_.push_back( v04 ) ;
		vtmp_.push_back( v00 ) ; vtmp_.push_back( v01 ) ; vtmp_.push_back( v05 ) ;
		vtmp_.push_back( v05 ) ; vtmp_.push_back( v04 ) ; vtmp_.push_back( v00 ) ;
		vtmp_.push_back( v07 ) ; vtmp_.push_back( v06 ) ; vtmp_.push_back( v02 ) ;
		vtmp_.push_back( v02 ) ; vtmp_.push_back( v03 ) ; vtmp_.push_back( v07 ) ;
		vtmp_.push_back( v00 ) ; vtmp_.push_back( v04 ) ; vtmp_.push_back( v07 ) ;
		vtmp_.push_back( v07 ) ; vtmp_.push_back( v03 ) ; vtmp_.push_back( v00 ) ;
		vtmp_.push_back( v05 ) ; vtmp_.push_back( v01 ) ; vtmp_.push_back( v02 ) ;
		vtmp_.push_back( v02 ) ; vtmp_.push_back( v06 ) ; vtmp_.push_back( v05 ) ;

		// convert to indexed vertices
		reduce() ;
	} else {
		// tetrahedron's vertices
		float3 v00 = {  m,  m,  m } ;
		float3 v01 = {  m, -m, -m } ;
		float3 v02 = { -m, -m,  m } ;
		float3 v03 = { -m,  m, -m } ;

		// tetrahedron's triangles
		pumpup( v00, v01, v02, ndiv_ ) ;
		pumpup( v00, v02, v03, ndiv_ ) ;
		pumpup( v00, v03, v01, ndiv_ ) ;
		pumpup( v03, v02, v01, ndiv_ ) ;

		reduce() ;
	}
}

void Sphere::pumpup( const float3& a, const float3& b, const float3& c, const int cdiv ) {
	if  ( cdiv>0 ) {
		float3 ab = V::unitV( .5f*( a+b ) ) ;
		float3 bc = V::unitV( .5f*( b+c ) ) ;
		float3 ca = V::unitV( .5f*( c+a ) ) ;

		pumpup(  a, ab, ca, cdiv-1 ) ;
		pumpup( ab,  b, bc, cdiv-1 ) ;
		pumpup( ca, bc,  c, cdiv-1 ) ;
		pumpup( ab, bc, ca, cdiv-1 ) ;
	} else {
		vtmp_.push_back( center_+radius_*a ) ;
		vtmp_.push_back( center_+radius_*b ) ;
		vtmp_.push_back( center_+radius_*c ) ;
	}
}

void Sphere::reduce() { // (SO #14396788)
	typedef std::pair<float3, uint> VIPair ;

	struct VIComp {
		bool operator () ( const VIPair& l, const VIPair& r ) const {
			if ( l.first.x != r.first.x )
				return l.first.x<r.first.x ;
			if ( l.first.y != r.first.y )
				return l.first.y<r.first.y ;
			if ( l.first.z != r.first.z )
				return l.first.z<r.first.z ;
			return false;
		}
	} ;

	std::set<VIPair, VIComp> irel ;
	std::vector<uint> itmp ;
	uint ice = 0 ;

	for ( const float3& vce : vtmp_ ) {
		std::set<VIPair>::iterator itor = irel.find( std::make_pair( vce, 0 ) ) ;

		if ( itor == irel.end() ) {
			irel.insert( std::make_pair( vce, ice ) ) ;
			itmp.push_back( ice++ ) ;
		} else
			itmp.push_back( itor->second ) ;
	}

	vces_.resize( irel.size() ) ;

	for ( std::set<VIPair>::iterator itor = irel.begin() ; itor != irel.end() ; itor++ )
		vces_[itor->second] = itor->first ;

	for ( uint i = 0 ; itmp.size()>i ; i+=3 )
		ices_.push_back( { itmp[i], itmp[i+1], itmp[i+2] } ) ;

	// vces_ = vtmp_ ;
	vtmp_.clear() ;
}
