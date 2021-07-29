#ifndef SPHERE_H
#define SPHERE_H

#include <vector_functions.h>
#include <vector_types.h>

#include "optics.h"
#include "thing.h"

class Sphere : public Thing {
	public:
		Sphere( const float3& center, const float radius, const Optics& optics, const bool bbox = false, const unsigned int ndiv = 6 ) ;
		~Sphere() noexcept ( false ) ;

	private:
		float  radius_ ;

		int ndiv_ ;

		std::vector<float3> vtmp_ ;

		void tetrahedron( const bool bbox ) ;
		void pumpup( const float3& a, const float3& b, const float3& c, const unsigned int ndiv ) ;
		void reduce() ;
} ;

#endif // SPHERE_H
