#include <set>
#include <vector>

#include <iostream>

struct uint3 {
	unsigned int x, y, z ;
} ;

struct float3 {
	float x, y, z ;
} ;

typedef std::pair<float3, unsigned int> VIRec ;

struct VICmp {
	bool operator () ( const VIRec& l, const VIRec& r ) const {
		if ( l.first.x != r.first.x )
			return l.first.x<r.first.x ;
		if ( l.first.y != r.first.y )
			return l.first.y<r.first.y ;
		if ( l.first.z != r.first.z )
			return l.first.z<r.first.z ;
		return false;
	}
} ;

std::vector<float3> vtmp_, vces_ ;
std::set<VIRec, VICmp> irel ;
std::vector<unsigned int> itmp ;
std::vector<uint3> ices_ ;

void reduce() { // (SO #14396788)
	unsigned int ice = 0 ;

	for ( const float3& vce : vtmp_ ) {
		std::set<VIRec>::iterator itor = irel.find( std::make_pair( vce, 0 ) ) ;

		if ( itor == irel.end() ) {
			irel.insert( std::make_pair( vce, ice ) ) ;
			itmp.push_back( ice++ ) ;
		} else
			itmp.push_back( itor->second ) ;
	}

	vces_.resize( irel.size() ) ;

	for ( std::set<VIRec>::iterator itor = irel.begin() ; itor != irel.end() ; itor++ )
		vces_[itor->second] = itor->first ;

	for ( unsigned int i = 0 ; itmp.size()>i ; i+=3 )
		ices_.push_back( { itmp[i], itmp[i+1], itmp[i+2] } ) ;

	vtmp_.clear() ;
}

int main() {
	vtmp_.push_back( {  0.f, 0.f, 0.f } ) ;
	vtmp_.push_back( {  1.f, 0.f, 0.f } ) ;
	vtmp_.push_back( {  .5f, 1.f, 0.f } ) ;
	vtmp_.push_back( {  1.f, 0.f, 0.f } ) ;
	vtmp_.push_back( {  2.f, 0.f, 0.f } ) ;
	vtmp_.push_back( { 1.5f, 1.f, 0.f } ) ;
	vtmp_.push_back( {  .5f, 1.f, 0.f } ) ;
	vtmp_.push_back( { 1.5f, 1.f, 0.f } ) ;
	vtmp_.push_back( {  1.f, 2.f, 0.f } ) ;
	vtmp_.push_back( {  .5f, 1.f, 0.f } ) ;
	vtmp_.push_back( { 1.5f, 1.f, 0.f } ) ;
	vtmp_.push_back( {  1.f, 0.f, 0.f } ) ;

	for ( int v = 0 ; vtmp_.size()>v ; v++ )
		printf( "{ %1.1f, %1.1f, %1.1f }\n", vtmp_[v].x, vtmp_[v].y, vtmp_[v].z ) ;

	printf( "\n" ) ;

	reduce() ;

	for ( std::set<VIRec>::iterator itor = irel.begin() ; itor != irel.end() ; itor++ ) {
		struct float3 v = itor->first ;
		printf( "{ %1.1f, %1.1f, %1.1f }, %d\n", v.x, v.y, v.z, itor->second) ;
	}

	printf( "\n" ) ;

	for ( int v = 0 ; vces_.size()>v ; v++ )
		printf( "{ %1.1f, %1.1f, %1.1f }\n", vces_[v].x, vces_[v].y, vces_[v].z ) ;

	printf( "\n" ) ;

	for ( int i = 0 ; itmp.size()>i ; i++ )
		printf( "%d\n", itmp[i] ) ; 

	printf( "\n" ) ;

	for ( int i = 0 ; ices_.size()>i ; i++ )
		printf( "{ %d, %d, %d }\n", ices_[i].x, ices_[i].y, ices_[i].z ) ;
}
