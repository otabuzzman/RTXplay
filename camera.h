#ifndef CAMERA_H
#define CAMERA_H

#include "v.h"

class Camera {
	public:
		Camera( const P eye, const P pat, const V vup, double fov, double aspratio, double aperture, double distance ) {
			w_ = unitV( eye-pat ) ;
			u_ = unitV( cross( vup, w_ ) ) ;
			v_ = cross( w_, u_ ) ;

			auto viewh = 2.*tan( deg2rad( fov )/2. ) ; // virtual viewport height
			auto vieww = aspratio*viewh ;              // virtual viewport width

			eye_  = eye ;
			lens_ = aperture/2. ;
			dist_ = distance ;
			vert_ = viewh*v_ ;
			hori_ = vieww*u_ ;
			bole_ = eye_-dist_*( hori_+vert_ )/2.-dist_*w_ ;
		}

		Ray ray( double s, double t ) const { V r = lens_*rndVin1disk() ; V o = r.x()*u_+r.y()*v_ ; return Ray( eye_+o, bole_+dist_*( s*hori_+t*vert_ )-eye_-o ) ; }

	private:
		V u_, v_, w_ ; // camera coordinate system
		P eye_ ;       // camera origin
		double lens_ ; // lens radius
		double dist_ ; // focus distance
		V vert_ ;      // virtual viewport height vector
		V hori_ ;      // virtual viewport width vector
		P bole_ ;      // bottom left viewport corner
} ;

#endif // CAMERA_H
