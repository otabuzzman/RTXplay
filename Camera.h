#ifndef CAMERA_H
#define CAMERA_H

#include "util.h"

class Camera {
    public:
        Camera() {
			r = 16./9. ;
			auto viewh = 2. ;                            // virtual viewport height
			auto vieww = r*viewh ;                       // virtual viewport width
			auto focll = 1. ;                            // focal length (projection point distance from ~plane)

			orig = P( 0, 0, 0 ) ;                        // camera origin
			hori = V( vieww, 0, 0 ) ;
			vert = V( 0, viewh, 0 ) ;
			bole = orig-hori/2-vert/2-V( 0, 0, focll ) ; // bottom left viewport corner
		}

		double ratio() const { return r ; }

		Ray ray( double u, double v ) const { return Ray( orig, bole+u*hori+v*vert-orig ) ; }

	private:
		double r ;
		P orig ;
		P bole ;
		V hori ;
		V vert ;
} ;

#endif