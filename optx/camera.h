#ifndef CAMERA_H
#define CAMERA_H

struct LpCamera {
	float3 eye ;
	float3 u ;
	float3 v ;
	float3 w ;
} ;

class Camera {
	public:
		Camera( const float3&  eye, const float3&  pat, const float3&  vup, const float fov, const float aspratio ) ;

		void set( LpCamera& camera ) const ;

	private:
		float3 eye_ ;
		float3 pat_ ;
		float3 vup_ ;
		float  fov_ ;
		float  aspratio_ ;
} ;

#endif // CAMERA_H
