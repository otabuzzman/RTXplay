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
			auto d = binding.normal+rndVon1sphere() ;
			if ( d.isnear0() )
				d = binding.normal ;

			sprayed = Ray( binding.p, d ) ;
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
			V d, u = unitV( ray.dir() ) ;
			double ctta = fmin( dot( -u, binding.normal ), 1. ) ;
			double stta = sqrt( 1.-ctta*ctta ) ;

			double ratio = binding.facing ? 1./index_ : index_ ;
			bool cannot = ratio*stta>1. ;

			if ( cannot || schlick( ctta, ratio )>rnd() )
				d = reflect( u, binding.normal ) ;
			else
				d = refract( u, binding.normal, ratio ) ;

			sprayed = Ray( binding.p, d ) ;
			attened = C( 1., 1., 1. ) ;

			return true ;
		}

	private:
		double index_ ; // refraction index

		// Schlick's reflectance approximation (chapter 10.4)
		static double schlick( double ctta, double ratio ) { auto r0 = ( 1-ratio )/( 1+ratio ) ; r0 = r0*r0 ; return r0+( 1-r0 )*pow( ( 1-ctta ), 5 ) ; }
} ;

#endif // OPTICS_H
