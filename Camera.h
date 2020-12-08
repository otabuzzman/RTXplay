#ifndef CAMERA_H
#define CAMERA_H

#include "V.h"

class Camera {
	public:
		Camera( P pos, P pov, V up, double fov, double aspect_ratio, double aperture, double focus_distance ) {
			auto viewh = 2.*tan( deg2rad( fov )/2 ) ;         // virtual viewport height
			auto vieww = aspect_ratio*viewh ;                 // virtual viewport width
			w_ = unitV( pos-pov ) ;
			u_ = unitV( cross( up, w_ ) ) ;
			v_ = cross( w_, u_ ) ;

			orig_ = pos ;                                     // camera origin
			hori_ = focus_distance*vieww*u_ ;
			vert_ = focus_distance*viewh*v_ ;
			bole_ = orig_-hori_/2-vert_/2-focus_distance*w_ ; // bottom left viewport corner
			lens_ = aperture/2 ;                              // lens radius
		}

		Ray ray( double s, double t ) const { V r = lens_*rndVin1disk() ; V o = r.x()*u_+r.y()*v_ ; return Ray( orig_+o, bole_+s*hori_+t*vert_-orig_-o ) ; }

	private:
		P orig_ ;
		P bole_ ;
		V hori_ ;
		V vert_ ;
		V u_, v_, w_ ;
		double lens_ ; // lens radius
} ;

#endif