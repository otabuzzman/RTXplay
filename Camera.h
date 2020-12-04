#ifndef CAMERA_H
#define CAMERA_H

#include "util.h"

class Camera {
    public:
        Camera( P pos, P pov, V up, double angle, double ratio ) {
			auto viewh = 2.*tan( deg2rad( angle )/2 ) ; // virtual viewport height
			auto vieww = ratio*viewh ;                  // virtual viewport width
			auto w = unitV( pos-pov ) ;
			auto u = unitV( cross( up, w ) ) ;
			auto v = cross( w, u ) ;

			orig = pos ;                              // camera origin
			hori = vieww*u ;
			vert = viewh*v ;
			bole = orig-hori/2-vert/2-w ;             // bottom left viewport corner
		}

		Ray ray( double s, double t ) const { return Ray( orig, bole+s*hori+t*vert-orig ) ; }

	private:
		P orig ;
		P bole ;
		V hori ;
		V vert ;
} ;

#endif