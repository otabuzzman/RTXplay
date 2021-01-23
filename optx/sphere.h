#ifndef SPHERE_H
#define SPHERE_H

class Sphere : public Thing {
	public:
		Sphere( const float3& center = { 0.f, 0.f, 0.f }, const float radius = 1.f, const bool bbox = false, const uint ndiv = 6 ) ;

	private:
		float3 center_ ;
		float  radius_ ;

		int ndiv_ ;

		std::vector<float3> vtmp_ ;

		void tetrahedron( const bool bbox ) ;
		void pumpup( const float3& a, const float3& b, const float3& c, const int ndiv ) ;
		void reduce() ;
} ;

#endif // SPHERE_H
