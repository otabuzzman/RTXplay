#ifndef OPTICS_H
#define OPTICS_H

#include "thing.h"

class Optics {
	public:
		virtual bool spray( const Ray& ray, const Binding& binding, C& attened, Ray& sprayed ) const = 0 ;
} ;

class Diffuse : public Optics {
	public:
		Diffuse( const C& albedo ) : albedo_( albedo ) {}

		virtual bool spray( const Ray& ray, const Binding& binding, C& attened, Ray& sprayed ) const override {
			auto dir = binding.normal+rndVon1sphere() ;
			if ( dir.isnear0() )
				dir = binding.normal ;

			sprayed = Ray( binding.p, dir ) ;
			attened = albedo_ ;

			return true ;
		}

	private:
		C albedo_ ;     // reflectivity
} ;

class Reflect : public Optics {
	public:
		Reflect( const C& albedo, double fuzz ) : albedo_( albedo ), fuzz_( fuzz ) {}

		virtual bool spray( const Ray& ray, const Binding& binding, C& attened, Ray& sprayed ) const override {
			V r = reflect( unitV( ray.dir() ), binding.normal ) ;
			sprayed = Ray( binding.p, r+fuzz_*rndVin1sphere() ) ;
			attened = albedo_ ;

			return ( dot( sprayed.dir(), binding.normal )>0 ) ;
		}

	private:
		C albedo_ ;     // reflectivity
		double fuzz_ ;  // matting
} ;

class Refract : public Optics {
	public:
		Refract( double index ) : index_( index ) {}

		virtual bool spray( const Ray& ray, const Binding& binding, C& attened, Ray& sprayed ) const override {
			V dir, d1V = unitV( ray.dir() ) ;
			double cos_theta = fmin( dot( -d1V, binding.normal ), 1. ) ;
			double sin_theta = sqrt( 1.-cos_theta*cos_theta ) ;

			double ratio = binding.facing ? 1./index_ : index_ ;
			bool cannot = ratio*sin_theta>1. ;

			if ( cannot || schlick( cos_theta, ratio )>rnd() )
				dir = reflect( d1V, binding.normal ) ;
			else
				dir = refract( d1V, binding.normal, ratio ) ;

			sprayed = Ray( binding.p, dir ) ;
			attened = C( 1., 1., 1. ) ;

			return true ;
		}

	private:
		double index_ ; // refraction index

		// Schlick's reflectance approximation (chapter 10.4)
		static double schlick( double cos_theta, double ratio ) { auto r0 = ( 1-ratio )/( 1+ratio ) ; r0 = r0*r0 ; return r0+( 1-r0 )*pow( ( 1-cos_theta ), 5 ) ; }
} ;

#endif // OPTICS_H
