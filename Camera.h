#ifndef CAMERA_H
#define CAMERA_H

#include "util.h"

class Camera {
	public:
		Camera( P pos, P pov, V up, double fov, double ratio, double aperture, double distance ) {
			auto viewh = 2.*tan( deg2rad( fov )/2 ) ; // virtual viewport height
			auto vieww = ratio*viewh ;                // virtual viewport width
			w = unitV( pos-pov ) ;
			u = unitV( cross( up, w ) ) ;
			v = cross( w, u ) ;

			orig = pos ;                              // camera origin
			hori = distance*vieww*u ;
			vert = distance*viewh*v ;
			bole = orig-hori/2-vert/2-distance*w ;    // bottom left viewport corner
			lnsr = aperture/2 ;                       // lens radius
		}

		Ray ray( double s, double t ) const { V r = lnsr*rndVin1disk() ; V o = r.x()*u+r.y()*v ; return Ray( orig+o, bole+s*hori+t*vert-orig-o ) ; }

	private:
		P orig ;
		P bole ;
		V hori ;
		V vert ;
		V u, v, w ;
		double lnsr ;
} ;

#endif