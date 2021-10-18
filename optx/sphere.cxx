// system includes
#include <set>
#include <vector>

// subsystem includes
// CUDA
#include <vector_types.h>

// local includes
#include "hoist.h"
#include "util.h"
#include "v.h"

// file specific includes
#include "sphere.h"

using V::operator+ ;
using V::operator* ;

Sphere::Sphere( const float radius, const unsigned int ndiv )
	: radius_( radius ), ndiv_( ndiv ) {

	tetrahedron() ;
}

void Sphere::tetrahedron() {
	// https://rechneronline.de/pi/tetrahedron.php
	// r = a/4*sqrt(6) | circumsphere radius r = 1
	// a = 4/sqrt(6)
	// m = a/4*sqrt(2)
	// precalculated value for unit tetrahedron
	float m = .57735026919f ; // midsphere radius

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

void Sphere::pumpup( const float3& a, const float3& b, const float3& c, const unsigned int cdiv ) {
	if  ( cdiv>0 ) {
		float3 ab = V::unitV( .5f*( a+b ) ) ;
		float3 bc = V::unitV( .5f*( b+c ) ) ;
		float3 ca = V::unitV( .5f*( c+a ) ) ;

		pumpup(  a, ab, ca, cdiv-1 ) ;
		pumpup( ab,  b, bc, cdiv-1 ) ;
		pumpup( ca, bc,  c, cdiv-1 ) ;
		pumpup( ab, bc, ca, cdiv-1 ) ;
	} else {
		vtmp_.push_back( radius_*a ) ;
		vtmp_.push_back( radius_*b ) ;
		vtmp_.push_back( radius_*c ) ;
	}
}

void Sphere::reduce() { // (SO #14396788)
	typedef std::pair<float3, unsigned int> VRec ;

	struct VCmp {
		bool operator() ( const VRec& l, const VRec& r ) const {
			if ( l.first.x != r.first.x )
				return l.first.x<r.first.x ;
			if ( l.first.y != r.first.y )
				return l.first.y<r.first.y ;
			if ( l.first.z != r.first.z )
				return l.first.z<r.first.z ;
			return false ;
		}
	} ;

	std::set<VRec, VCmp> irel ;
	std::vector<unsigned int> itmp ;
	unsigned int ice = 0 ;

	for ( const float3& vce : vtmp_ ) {
		std::set<VRec>::iterator vrec = irel.find( std::make_pair( vce, 0 ) ) ;

		if ( vrec == irel.end() ) {
			irel.insert( std::make_pair( vce, ice ) ) ;
			itmp.push_back( ice++ ) ;
		} else
			itmp.push_back( vrec->second ) ;
	}

	vces_.resize( irel.size() ) ;

	for ( std::set<VRec>::iterator vrec = irel.begin() ; vrec != irel.end() ; vrec++ )
		vces_[vrec->second] = vrec->first ;

	for ( unsigned int i = 0 ; itmp.size()>i ; i+=3 )
		ices_.push_back( { itmp[i], itmp[i+1], itmp[i+2] } ) ;

	vtmp_.clear() ;
}

#ifdef MAIN

int main( const int argc, const char** argv ) {
	float radius = 1.f ;
	unsigned int ndiv = 6 ;

	if ( argc>1 ) sscanf( argv[1], "%f", &radius ) ;
	if ( argc>2 ) sscanf( argv[2], "%u", &ndiv ) ;
	Sphere sphere( radius, ndiv ) ;

	std::cout << "# sphere approximation by pumped-up tetrahedron:"              << std::endl ;
	std::cout << "# " << ndiv << " times recursive triangle surface subdivision" << std::endl ;

	for ( auto v : sphere.vces() )
		printf( "v %f %f %f\n", v.x, v.y, v.z ) ;
	std::cout << "# " << sphere.vces().size() << " vertices"  << std::endl ;

	for ( auto i : sphere.ices() )
		printf( "f %d %d %d\n", i.x+1, i.y+1, i.z+1 ) ;
	std::cout << "# " << sphere.ices().size() << " triangles" << std::endl ;

	std::cout << std::endl ;

	return 0 ;
}

#endif // MAIN
