#ifndef SPHERE_H
#define SPHERE_H

#include <vector_types.h>

#include "thing.h"

class Sphere : public Thing {
	public:
		Sphere( const float radius, const bool bbox = false, const unsigned int ndiv = 6 ) ;
		~Sphere() noexcept ( false ) ;

	private:
		float radius_ ;
		int   ndiv_ ;

		std::vector<float3> vces_ ; // sphere's unique vertices...
		std::vector<uint3>  ices_ ; // ...as indexed triangles

		std::vector<float3> vtmp_ ;

		void tetrahedron( const bool bbox ) ;
		void pumpup( const float3& a, const float3& b, const float3& c, const unsigned int ndiv ) ;
		void reduce() ;

		void set( const std::vector<float3>& vces ) ;
		void set( const std::vector<uint3>&  ices ) ;
} ;

#endif // SPHERE_H
