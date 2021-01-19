#ifndef CAMERA_H
#define CAMERA_H

struct LpCamera {
	float3 u ;
	float3 v ;
	float3 w ;
	float3 eye ;
	float  lens ;
	float  dist ;
	float3 hvec ;
	float3 wvec ;
} ;

class Camera {
	public:
		Camera( const float3&  eye, const float3&  pat, const float3&  vup, const float fov, const float aspratio, const float aperture, const float distance ) ;

		void set( LpCamera& camera ) const ;

	private:
		float3 eye_ ;
		float3 pat_ ;
		float3 vup_ ;
		float  fov_ ;
		float  aspratio_ ;
		float  aperture_ ;
		float  distance_ ;
} ;

#endif // CAMERA_H
