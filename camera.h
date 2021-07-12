#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

class Camera {
	public:
		Camera() {} ;

		void set( const P& eye, const P& pat, const V& vup, const double fov, const double aspratio, const double aperture, const double distance ) {
			eye_       = eye ;
			aperture_  = aperture ;

			w_ = unitV( eye-pat ) ;
			u_ = unitV( cross( vup, w_ ) ) ;
			v_ = cross( w_, u_ ) ;

			auto hlen = 2.*tan( .5*fov*kPi/180. ) ; // virtual viewport height
			auto wlen = hlen*aspratio ;             // virtual viewport width
			hvec_ = distance*hlen/2.*v_ ;
			wvec_ = distance*wlen/2.*u_ ;
		}

		Ray ray( const double s, const double t ) const {
			V r = aperture_/2.*rndVin1disk() ;
			V o = eye_+r.x()*u_+r.y()*v_ ;

			return Ray( o, s*wvec_+t*hvec_-o ) ;
		}

	private:
		P eye_ ;           // camera origin
		double aperture_ ;

		V u_, v_, w_ ;     // camera coordinate system
		V hvec_ ;          // virtual viewport height vector
		V wvec_ ;          // virtual viewport width vector
} ;

#endif // CAMERA_H
