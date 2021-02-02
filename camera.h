#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

class Camera {
	public:
		Camera() {} ;

		void set( const P& eye, const P& pat, const V& vup, const double fov, const double aspratio, const double aperture, const double distance ) {
			w_ = unitV( eye-pat ) ;
			u_ = unitV( cross( vup, w_ ) ) ;
			v_ = cross( w_, u_ ) ;

			auto hlen = 2.*tan( .5*fov*kPi/180. ) ; // virtual viewport height
			auto wlen = hlen*aspratio ;             // virtual viewport width

			eye_  = eye ;
			lens_ = aperture/2. ;
			dist_ = distance ;
			hvec_ = hlen*v_ ;
			wvec_ = wlen*u_ ;
		}

		Ray ray( const double s, const double t ) const {
			V r = lens_*rndVin1disk() ;
			V o = eye_+r.x()*u_+r.y()*v_ ;

			return Ray( o, dist_*( s*wvec_+t*hvec_ )-o ) ;
		}

	private:
		V u_, v_, w_ ; // camera coordinate system
		P eye_ ;       // camera origin
		double lens_ ; // lens radius
		double dist_ ; // focus distance
		V hvec_ ;      // virtual viewport height vector
		V wvec_ ;      // virtual viewport width vector
} ;

#endif // CAMERA_H
