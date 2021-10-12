#ifndef SPHERE_H
#define SPHERE_H

// system includes
#include <vector>

// subsystem includes
// CUDA
#include <vector_types.h>

// local includes
// none

// file specific includes
// none

class Sphere {
	public:
		Sphere( const float radius = 1.f, const unsigned int ndiv = 6 ) ;

		const std::vector<float3>& vces() const { return vces_ ; } ;
		const std::vector<uint3>&  ices() const { return ices_ ; } ;

	private:
		float radius_ ;
		int   ndiv_   ;

		std::vector<float3> vces_ ; // sphere's unique vertices...
		std::vector<uint3>  ices_ ; // ...as indexed triangles

		std::vector<float3> vtmp_ ;

		void tetrahedron() ;
		void pumpup( const float3& a, const float3& b, const float3& c, const unsigned int ndiv ) ;
		void reduce() ;
} ;

#endif // SPHERE_H
