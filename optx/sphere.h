#include <vector.h>

#include <vector_functions.h>
#include <vector_types.h>

#include "v.h"
#include "util.h"

#include "thing.h"
#include "sphere.h"

class Sphere : public Thing {
	public:
		Sphere( const float3& center = { 0.f, 0.f, 0.f }, const float radius = 1.f, const bool bbox = false ) ;

	private:
		float3 center_ ;
		float  radius_ ;

		void icosahedron( const int ndiv ) const ;
		void divide( const float3& a, const float3& b, const float3& c, const int ndiv ) const ;
} ;

#endif // SPHERE_H
