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
			auto d = binding.normal + rndVon1sphere() ;
			if ( d.isnear0() )
				d = binding.normal ;

			sprayed = Ray( binding.p, d ) ;
			attened = albedo_ ;

			return true ;
		}

	public:
		C albedo_ ;    // reflectivity
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

	public:
		C albedo_ ;    // reflectivity
		double fuzz_ ; // fuzz
} ;

class Refract : public Optics {
	public:
		Refract( double refraction_index ) : refraction_index_( refraction_index ) {}

		virtual bool spray( const Ray& ray, const Binding& binding, C& attened, Ray& sprayed ) const override {
			attened = C( 1., 1., 1. ) ;
			V d, u = unitV( ray.dir() ) ;
			double ctta = fmin( dot( -u, binding.normal ), 1. ) ;
			double stta = sqrt( 1.-ctta*ctta ) ;

			double refraction_ratio = binding.facing ? 1./refraction_index_ : refraction_index_ ;
			bool cannot = refraction_ratio*stta>1. ;

			if ( cannot || schlick( ctta, refraction_ratio )>rnd() )
				d = reflect( u, binding.normal ) ;
			else
				d = refract( u, binding.normal, refraction_ratio ) ;

			sprayed = Ray( binding.p, d ) ;
			return true ;
		}

	private:
		double refraction_index_ ;

		// Schlick's reflectance approximation (chapter 10.4)
		static double schlick( double ctta, double rrat ) { auto r0 = ( 1-rrat )/( 1+rrat ) ; r0 = r0*r0 ; return r0+( 1-r0 )*pow( ( 1-ctta ), 5 ) ; }
} ;

#endif // OPTICS_H
