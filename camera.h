#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

class Camera {
	public:
		Camera() {} ;

		void set( const P& eye, const P& pat, const V& vup, const double fov, const double aspratio, const double aperture, const double fostance ) {
			eye_       = eye ;
			aperture_  = aperture ;

			w_ = unitV( eye-pat ) ;
			u_ = unitV( cross( vup, w_ ) ) ;
			v_ = cross( w_, u_ ) ;

			auto h = 2.*tan( .5*fov*kPi/180. ) ; // focus plane (virtual viewport) height
			auto w = h*aspratio ;                // focus plane width
			hvec_ = fostance*h/2.*v_ ; // focus plane height vector
			wvec_ = fostance*w/2.*u_ ; // focus plane width vector
			dvec_ = fostance     *w_ ; // lens/ focus plane distance vector (focus distance)
		}

		Ray ray( const double s, const double t ) const {
			// defocus blur (depth of field)
			V r = aperture_/2.*rndVin1disk() ;
			V o = r.x()*u_+r.y()*v_ ;

			return Ray( eye_+o, s*wvec_+t*hvec_-dvec_-o ) ;
		}

	private:
		P eye_ ;           // camera origin
		double aperture_ ;

		V u_, v_, w_ ;     // camera coordinate system
		V hvec_ ;          // virtual viewport height vector
		V wvec_ ;          // virtual viewport width vector
		V dvec_ ;          // lens/ focus plane fostance vector (focus fostance)
} ;

#endif // CAMERA_H
