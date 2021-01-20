#ifndef SPHERE_H
#define SPHERE_H

class Sphere : public Thing {
	public:
		Sphere( const float3& center = { 0.f, 0.f, 0.f }, const float radius = 1.f, const bool bbox = false ) ;

	private:
		float3 center_ ;
		float  radius_ ;

		void icosahedron( const int ndiv, const bool bbox ) ;
		void divide( const float3& a, const float3& b, const float3& c, const int ndiv ) ;
} ;

#endif // SPHERE_H
